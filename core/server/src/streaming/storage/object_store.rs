/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

//! Pluggable object storage abstraction.
//!
//! `ObjectStorage` is a small async trait that backs the milestone's "S3 as
//! optional primary storage" feature. Phase 2+ persistence sites consume an
//! `Arc<dyn ObjectStorage>`; the choice is made at boot from `[system.storage]`.
//!
//! Backends:
//!
//! | Backend            | Where it lives                                   |
//! |--------------------|--------------------------------------------------|
//! | `CompioFsStorage`  | this file — passthrough to the local filesystem  |
//! | `InMemoryStorage`  | this file (cfg(test)) — HashMap-backed for tests |
//! | `S3Storage`        | added in Phase 1b (rusty-s3 + cyper)             |

use std::ops::Range;

use async_trait::async_trait;
use bytes::Bytes;
use compio::fs::OpenOptions;
use compio::io::{AsyncReadAtExt, AsyncWriteAtExt};
use err_trail::ErrContext;
use iggy_common::IggyError;

const COMPONENT: &str = "STREAMING_OBJECT_STORAGE";

/// Lightweight metadata returned by [`ObjectStorage::head`] and
/// [`ObjectStorage::list_prefix`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectMeta {
    pub key: String,
    pub size: u64,
    /// Backend-supplied entity tag, when available. The fs backend leaves
    /// this `None`; S3 returns the server-assigned ETag.
    pub etag: Option<String>,
}

/// Pluggable byte-addressable storage.
///
/// Designed to be cheap to share across tasks within a compio shard:
/// implementations are stateless or hold internal handles behind their own
/// synchronization, and callers pass `Rc<dyn ObjectStorage>` around freely.
/// The trait is `?Send` because compio's per-thread io_uring driver yields
/// non-`Send` futures; `Arc<dyn ObjectStorage>` across shards isn't a
/// supported pattern (each shard owns its own instance).
#[async_trait(?Send)]
pub trait ObjectStorage: std::fmt::Debug {
    /// Write `bytes` at `key`, replacing any existing object atomically from
    /// the reader's perspective.
    async fn put(&self, key: &str, bytes: Bytes) -> Result<(), IggyError>;

    /// Conditional create: succeeds only when no object exists at `key`.
    /// Returns `IggyError::CannotOverwriteFile` if the object already exists.
    /// Used by Phase 8 fencing.
    async fn put_if_absent(&self, key: &str, bytes: Bytes) -> Result<(), IggyError>;

    /// Open a streaming write. The handle accumulates bytes via `upload_part`
    /// and seals via `complete` (or discards via `abort`).
    async fn put_multipart(&self, key: &str) -> Result<Box<dyn MultipartHandle>, IggyError>;

    /// Read the byte range `[range.start, range.end)`. Returns exactly
    /// `range.end - range.start` bytes on success.
    async fn get_range(&self, key: &str, range: Range<u64>) -> Result<Bytes, IggyError>;

    /// Object metadata. Errors with `CannotReadFileMetadata` if absent.
    async fn head(&self, key: &str) -> Result<ObjectMeta, IggyError>;

    /// Enumerate every object whose key begins with `prefix`. Order is
    /// lexicographic on the key. The fs backend treats `prefix` as a
    /// directory path and walks it recursively.
    async fn list_prefix(&self, prefix: &str) -> Result<Vec<ObjectMeta>, IggyError>;

    /// Delete a single object. Idempotent — deleting a missing key succeeds.
    async fn delete(&self, key: &str) -> Result<(), IggyError>;

    /// Delete many objects. Backends with batch APIs (S3 `DeleteObjects`)
    /// override this; the default is a per-key loop.
    async fn delete_many(&self, keys: &[&str]) -> Result<(), IggyError> {
        for k in keys {
            self.delete(k).await?;
        }
        Ok(())
    }
}

/// In-progress streaming write returned by [`ObjectStorage::put_multipart`].
///
/// The handle is consumed by either `complete` (seal) or `abort` (discard);
/// the `self: Box<Self>` bounds make the consumption explicit at the call
/// site. Dropping the handle without calling either is a programming error
/// — backends best-effort abort on drop where possible, but callers should
/// not rely on it. Like [`ObjectStorage`], the trait is `?Send` because the
/// compio io_uring driver yields non-`Send` futures.
#[async_trait(?Send)]
pub trait MultipartHandle {
    /// Append `bytes` to the in-progress upload.
    async fn upload_part(&mut self, bytes: Bytes) -> Result<(), IggyError>;

    /// Seal the upload. Returns the backend-assigned final identifier (ETag
    /// for S3; empty string for fs).
    async fn complete(self: Box<Self>) -> Result<String, IggyError>;

    /// Discard the upload and release any backend-side resources.
    async fn abort(self: Box<Self>) -> Result<(), IggyError>;
}

// =====================================================================
// CompioFsStorage — local filesystem
// =====================================================================

/// Filesystem-backed `ObjectStorage`. Keys are filesystem paths.
///
/// This is a thin wrapper over the same compio fs APIs the legacy
/// `PersisterKind::File` already uses; it does not introduce new I/O
/// behavior and is byte-compatible with existing fs-mode deployments.
#[derive(Debug, Default, Clone)]
pub struct CompioFsStorage;

#[async_trait(?Send)]
impl ObjectStorage for CompioFsStorage {
    async fn put(&self, key: &str, bytes: Bytes) -> Result<(), IggyError> {
        let mut file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(key)
            .await
            .error(|e: &std::io::Error| format!("{COMPONENT} (error: {e}) - put open: {key}"))
            .map_err(|_| IggyError::CannotOverwriteFile)?;
        file.write_all_at(bytes.to_vec(), 0)
            .await
            .0
            .error(|e: &std::io::Error| format!("{COMPONENT} (error: {e}) - put write: {key}"))
            .map_err(|_| IggyError::CannotWriteToFile)?;
        Ok(())
    }

    async fn put_if_absent(&self, key: &str, bytes: Bytes) -> Result<(), IggyError> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(key)
            .await
            .error(|e: &std::io::Error| {
                format!("{COMPONENT} (error: {e}) - put_if_absent open: {key}")
            })
            .map_err(|_| IggyError::CannotOverwriteFile)?;
        file.write_all_at(bytes.to_vec(), 0)
            .await
            .0
            .error(|e: &std::io::Error| {
                format!("{COMPONENT} (error: {e}) - put_if_absent write: {key}")
            })
            .map_err(|_| IggyError::CannotWriteToFile)?;
        Ok(())
    }

    async fn put_multipart(&self, key: &str) -> Result<Box<dyn MultipartHandle>, IggyError> {
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(key)
            .await
            .error(|e: &std::io::Error| {
                format!("{COMPONENT} (error: {e}) - put_multipart open: {key}")
            })
            .map_err(|_| IggyError::CannotOverwriteFile)?;
        Ok(Box::new(FsMultipart {
            key: key.to_owned(),
            file: Some(file),
            offset: 0,
        }))
    }

    async fn get_range(&self, key: &str, range: Range<u64>) -> Result<Bytes, IggyError> {
        let len = range
            .end
            .checked_sub(range.start)
            .ok_or(IggyError::CannotReadFile)? as usize;
        let file = OpenOptions::new()
            .read(true)
            .open(key)
            .await
            .error(|e: &std::io::Error| format!("{COMPONENT} (error: {e}) - get_range open: {key}"))
            .map_err(|_| IggyError::CannotReadFile)?;
        let (result, buf): (std::io::Result<()>, Vec<u8>) =
            file.read_exact_at(vec![0u8; len], range.start).await.into();
        result
            .error(|e: &std::io::Error| format!("{COMPONENT} (error: {e}) - get_range read: {key}"))
            .map_err(|_| IggyError::CannotReadFile)?;
        Ok(Bytes::from(buf))
    }

    async fn head(&self, key: &str) -> Result<ObjectMeta, IggyError> {
        let meta = std::fs::metadata(key)
            .error(|e: &std::io::Error| format!("{COMPONENT} (error: {e}) - head: {key}"))
            .map_err(|_| IggyError::CannotReadFileMetadata)?;
        Ok(ObjectMeta {
            key: key.to_owned(),
            size: meta.len(),
            etag: None,
        })
    }

    async fn list_prefix(&self, prefix: &str) -> Result<Vec<ObjectMeta>, IggyError> {
        // NOTE: blocking std::fs walk. Phase 4 replaces production callers
        // with fs-mode-aware async paths; for the fs backend itself this is
        // unchanged from current iggy behavior.
        let mut out = Vec::new();
        walk_fs(prefix, &mut out)?;
        out.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(out)
    }

    async fn delete(&self, key: &str) -> Result<(), IggyError> {
        match compio::fs::remove_file(key).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(IggyError::CannotDeleteFile)
                .error(|_: &IggyError| format!("{COMPONENT} (error: {e}) - delete: {key}")),
        }
    }
}

fn walk_fs(prefix: &str, out: &mut Vec<ObjectMeta>) -> Result<(), IggyError> {
    let entries = match std::fs::read_dir(prefix) {
        Ok(it) => it,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(IggyError::CannotReadFile).error(|_: &IggyError| {
                format!("{COMPONENT} (error: {e}) - list_prefix: {prefix}")
            });
        }
    };
    for entry in entries {
        let entry = entry
            .error(|e: &std::io::Error| format!("{COMPONENT} (error: {e}) - read_dir: {prefix}"))
            .map_err(|_| IggyError::CannotReadFile)?;
        let path = entry.path();
        let path_str = path.to_string_lossy().into_owned();
        let ftype = entry
            .file_type()
            .error(|e: &std::io::Error| format!("{COMPONENT} (error: {e}) - file_type: {path_str}"))
            .map_err(|_| IggyError::CannotReadFileMetadata)?;
        if ftype.is_dir() {
            walk_fs(&path_str, out)?;
        } else if ftype.is_file() {
            let meta = entry
                .metadata()
                .error(|e: &std::io::Error| format!("{COMPONENT} (error: {e}) - meta: {path_str}"))
                .map_err(|_| IggyError::CannotReadFileMetadata)?;
            out.push(ObjectMeta {
                key: path_str,
                size: meta.len(),
                etag: None,
            });
        }
    }
    Ok(())
}

struct FsMultipart {
    key: String,
    file: Option<compio::fs::File>,
    offset: u64,
}

#[async_trait(?Send)]
impl MultipartHandle for FsMultipart {
    async fn upload_part(&mut self, bytes: Bytes) -> Result<(), IggyError> {
        let file = self.file.as_mut().ok_or_else(|| {
            IggyError::CannotWriteToFile.error_context(format!(
                "{COMPONENT} - upload_part on already-finished handle: {}",
                self.key,
            ))
        })?;
        let len = bytes.len() as u64;
        let key = self.key.clone();
        let offset = self.offset;
        file.write_all_at(bytes.to_vec(), offset)
            .await
            .0
            .error(|e: &std::io::Error| {
                format!("{COMPONENT} (error: {e}) - upload_part write: {key}")
            })
            .map_err(|_| IggyError::CannotWriteToFile)?;
        self.offset += len;
        Ok(())
    }

    async fn complete(mut self: Box<Self>) -> Result<String, IggyError> {
        // Drop the file handle; bytes already on disk.
        let _ = self.file.take();
        Ok(String::new())
    }

    async fn abort(mut self: Box<Self>) -> Result<(), IggyError> {
        let _ = self.file.take();
        match compio::fs::remove_file(&self.key).await {
            Ok(()) | Err(_) => Ok(()),
        }
    }
}

/// Tiny shim so call sites can attach context to `IggyError` after the fact
/// without introducing a heavier error-wrapping layer in this module.
trait IggyErrorContext {
    fn error_context(self, msg: String) -> Self;
}

impl IggyErrorContext for IggyError {
    fn error_context(self, msg: String) -> Self {
        tracing::warn!("{msg}");
        self
    }
}

// =====================================================================
// BufferedMultipartWriter — coalesces small flushes to S3 part minimums
// =====================================================================

use std::rc::Rc;

/// Buffers writes until they reach `part_size`, then uploads each chunk as
/// one part of an in-progress multipart write.
///
/// This adapter is what makes iggy's typical sub-MiB flush sizes compatible
/// with AWS S3's hard 5 MiB minimum part size (except the final part).
/// `seal()` flushes any residual buffer as the last part. As a small-segment
/// optimization, if no parts have been uploaded yet AND the residual buffer
/// is below the 5 MiB part minimum, the multipart is aborted and the buffer
/// is written via a single `put` instead — avoiding a CompleteMultipartUpload
/// that would otherwise fail with `EntityTooSmall`.
pub struct BufferedMultipartWriter {
    storage: Rc<dyn ObjectStorage>,
    key: String,
    handle: Option<Box<dyn MultipartHandle>>,
    buffer: bytes::BytesMut,
    part_size: usize,
    parts_uploaded: u32,
}

/// AWS S3 minimum size for any non-final multipart part.
pub const S3_MIN_PART_SIZE: usize = 5 * 1024 * 1024;

impl BufferedMultipartWriter {
    /// Open a buffered multipart write at `key`. `part_size` must be at
    /// least [`S3_MIN_PART_SIZE`] (5 MiB) — this is enforced at config-load
    /// time but re-asserted here for direct callers.
    pub async fn open(
        storage: Rc<dyn ObjectStorage>,
        key: &str,
        part_size: usize,
    ) -> Result<Self, IggyError> {
        debug_assert!(
            part_size >= S3_MIN_PART_SIZE,
            "BufferedMultipartWriter::open: part_size {} below S3 5 MiB minimum",
            part_size,
        );
        let handle = storage.put_multipart(key).await?;
        Ok(Self {
            storage,
            key: key.to_owned(),
            handle: Some(handle),
            buffer: bytes::BytesMut::with_capacity(part_size),
            part_size,
            parts_uploaded: 0,
        })
    }

    /// Append bytes to the in-progress upload, flushing whole parts to the
    /// backend as the buffer crosses `part_size`.
    pub async fn append(&mut self, bytes: &[u8]) -> Result<(), IggyError> {
        self.buffer.extend_from_slice(bytes);
        while self.buffer.len() >= self.part_size {
            let chunk = self.buffer.split_to(self.part_size).freeze();
            let handle = self.handle.as_mut().ok_or(IggyError::CannotAppendToFile)?;
            handle.upload_part(chunk).await?;
            self.parts_uploaded += 1;
        }
        Ok(())
    }

    /// Total bytes appended so far (across uploaded parts and the buffer).
    pub fn buffered_bytes(&self) -> usize {
        self.buffer.len()
    }

    /// Number of complete parts uploaded so far.
    pub fn parts_uploaded(&self) -> u32 {
        self.parts_uploaded
    }

    /// Seal the upload. Returns the backend etag, or empty string if the
    /// small-segment fallback path was taken.
    pub async fn seal(mut self) -> Result<String, IggyError> {
        // Small-segment optimization: nothing has been parted out yet AND the
        // residual is below S3's minimum — abort multipart, single PUT.
        if self.parts_uploaded == 0 && self.buffer.len() < S3_MIN_PART_SIZE {
            if let Some(handle) = self.handle.take() {
                handle.abort().await?;
            }
            let bytes = self.buffer.freeze();
            self.storage.put(&self.key, bytes).await?;
            return Ok(String::new());
        }

        // Normal path: flush residual as the final part, then complete.
        let mut handle = self.handle.take().ok_or(IggyError::CannotAppendToFile)?;
        if !self.buffer.is_empty() {
            let final_part = std::mem::take(&mut self.buffer).freeze();
            handle.upload_part(final_part).await?;
        }
        handle.complete().await
    }

    /// Discard the upload and any backend state.
    pub async fn abort(mut self) -> Result<(), IggyError> {
        if let Some(handle) = self.handle.take() {
            handle.abort().await?;
        }
        Ok(())
    }
}

// =====================================================================
// S3Storage — rusty-s3 + cyper, behind cargo features = ["object-storage"]
// =====================================================================

#[cfg(feature = "object-storage")]
mod s3 {
    use super::*;
    use crate::configs::system::ObjectStorageConfig;
    use rusty_s3::actions::S3Action;
    use rusty_s3::{Bucket, Credentials, UrlStyle};
    use std::time::Duration;

    /// Time-to-live for presigned URLs. Each S3 call signs a short-lived
    /// URL and dispatches it via cyper. A few minutes is plenty for the
    /// per-call latency we observe in practice.
    const PRESIGN: Duration = Duration::from_secs(300);

    /// AWS S3 client built on rusty-s3 (sans-IO SigV4 + request building)
    /// and cyper (compio HTTP client, rustls TLS). Phase 0 spike validated
    /// this combination against AWS S3 in account 208116833703 / us-east-1.
    pub struct S3Storage {
        bucket: Bucket,
        creds: Credentials,
        http: cyper::Client,
        prefix: String,
    }

    impl std::fmt::Debug for S3Storage {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("S3Storage")
                .field("bucket", &self.bucket.name())
                .field("region", &self.bucket.region())
                .field("prefix", &self.prefix)
                .finish_non_exhaustive()
        }
    }

    impl S3Storage {
        /// Build from `[system.storage.object]` config. Credentials are read
        /// from config first; callers wanting env-var or profile resolution
        /// should populate the config struct upstream of this constructor
        /// (see `crate::bootstrap::resolve_object_storage`).
        pub fn from_config(config: &ObjectStorageConfig) -> Result<Self, IggyError> {
            if !config.service.eq_ignore_ascii_case("s3") {
                tracing::warn!(
                    "system.storage.object.service = {:?}; only \"s3\" is recognized",
                    config.service,
                );
            }
            if config.bucket.is_empty() {
                return Err(IggyError::CannotOverwriteFile);
            }
            let endpoint = endpoint_url(config)?;
            let url_style = if config.endpoint.is_empty() {
                UrlStyle::VirtualHost
            } else {
                UrlStyle::Path
            };
            let bucket = Bucket::new(
                endpoint,
                url_style,
                config.bucket.clone(),
                config.region.clone(),
            )
            .error(|e: &rusty_s3::BucketError| {
                format!("{COMPONENT} (error: {e}) - construct rusty-s3 Bucket")
            })
            .map_err(|_| IggyError::CannotOverwriteFile)?;
            let creds = if config.access_key_id.is_empty() && config.secret_access_key.is_empty() {
                return Err(IggyError::CannotOverwriteFile);
            } else {
                Credentials::new(
                    config.access_key_id.clone(),
                    config.secret_access_key.clone(),
                )
            };
            Ok(Self {
                bucket,
                creds,
                http: cyper::Client::new(),
                prefix: config.prefix.trim_end_matches('/').to_owned(),
            })
        }

        fn full_key(&self, key: &str) -> String {
            if self.prefix.is_empty() {
                key.to_owned()
            } else {
                format!("{}/{}", self.prefix, key.trim_start_matches('/'))
            }
        }

        async fn execute(
            &self,
            req: cyper::Request,
            op: &str,
        ) -> Result<cyper::Response, IggyError> {
            self.http
                .execute(req)
                .await
                .error(|e: &cyper::Error| format!("{COMPONENT} (error: {e}) - {op}"))
                .map_err(|_| IggyError::CannotWriteToFile)
        }
    }

    fn endpoint_url(config: &ObjectStorageConfig) -> Result<url::Url, IggyError> {
        let raw = if config.endpoint.is_empty() {
            // AWS default. us-east-1 has a region-less endpoint historically,
            // but the regional form works everywhere.
            format!("https://s3.{}.amazonaws.com", config.region)
        } else {
            config.endpoint.clone()
        };
        raw.parse::<url::Url>()
            .error(|e: &url::ParseError| {
                format!("{COMPONENT} (error: {e}) - parse endpoint: {raw}")
            })
            .map_err(|_| IggyError::CannotOverwriteFile)
    }

    fn require_2xx(resp: &cyper::Response, op: &str) -> Result<(), IggyError> {
        if resp.status().is_success() {
            return Ok(());
        }
        tracing::warn!(
            target: "object_store",
            "{COMPONENT} - {op}: HTTP {}",
            resp.status().as_u16(),
        );
        // Caller picks the best IggyError variant; this is just the gate.
        Err(IggyError::CannotWriteToFile)
    }

    fn strip_etag(value: &str) -> String {
        value.trim_matches('"').to_owned()
    }

    #[async_trait(?Send)]
    impl ObjectStorage for S3Storage {
        async fn put(&self, key: &str, bytes: Bytes) -> Result<(), IggyError> {
            let key = self.full_key(key);
            let action = self.bucket.put_object(Some(&self.creds), &key);
            let url = action.sign(PRESIGN);
            let req = self
                .http
                .put(url.as_str())
                .map_err(|_| IggyError::CannotWriteToFile)?
                .body(bytes.to_vec())
                .build();
            let resp = self.execute(req, "PUT").await?;
            require_2xx(&resp, "PUT")
        }

        async fn put_if_absent(&self, key: &str, bytes: Bytes) -> Result<(), IggyError> {
            let key = self.full_key(key);
            let action = self.bucket.put_object(Some(&self.creds), &key);
            let url = action.sign(PRESIGN);
            let req = self
                .http
                .put(url.as_str())
                .map_err(|_| IggyError::CannotWriteToFile)?
                .header("If-None-Match", "*")
                .map_err(|_| IggyError::CannotWriteToFile)?
                .body(bytes.to_vec())
                .build();
            let resp = self.execute(req, "PUT-if-none-match").await?;
            if resp.status().as_u16() == 412 {
                return Err(IggyError::CannotOverwriteFile);
            }
            require_2xx(&resp, "PUT-if-none-match")
        }

        async fn put_multipart(&self, key: &str) -> Result<Box<dyn MultipartHandle>, IggyError> {
            let key = self.full_key(key);
            let create = self.bucket.create_multipart_upload(Some(&self.creds), &key);
            let url = create.sign(PRESIGN);
            let req = self
                .http
                .post(url.as_str())
                .map_err(|_| IggyError::CannotWriteToFile)?
                .build();
            let resp = self.execute(req, "CreateMultipartUpload").await?;
            require_2xx(&resp, "CreateMultipartUpload")?;
            let body = resp
                .text()
                .await
                .map_err(|_| IggyError::CannotWriteToFile)?;
            let parsed =
                rusty_s3::actions::CreateMultipartUpload::parse_response(&body).map_err(|e| {
                    tracing::warn!("{COMPONENT} (error: {e}) - parse CreateMultipartUpload");
                    IggyError::CannotWriteToFile
                })?;
            Ok(Box::new(S3Multipart {
                bucket: self.bucket.clone(),
                creds: self.creds.clone(),
                http: self.http.clone(),
                key,
                upload_id: parsed.upload_id().to_owned(),
                parts: Vec::new(),
                next_part: 1,
            }))
        }

        async fn get_range(&self, key: &str, range: Range<u64>) -> Result<Bytes, IggyError> {
            let key = self.full_key(key);
            let action = self.bucket.get_object(Some(&self.creds), &key);
            let url = action.sign(PRESIGN);
            // S3 Range header is inclusive on both ends.
            let range_header = format!("bytes={}-{}", range.start, range.end.saturating_sub(1),);
            let req = self
                .http
                .get(url.as_str())
                .map_err(|_| IggyError::CannotReadFile)?
                .header("Range", range_header)
                .map_err(|_| IggyError::CannotReadFile)?
                .build();
            let resp = self.execute(req, "GET-range").await?;
            if !resp.status().is_success() {
                return Err(IggyError::CannotReadFile);
            }
            resp.bytes().await.map_err(|_| IggyError::CannotReadFile)
        }

        async fn head(&self, key: &str) -> Result<ObjectMeta, IggyError> {
            let full = self.full_key(key);
            let action = self.bucket.head_object(Some(&self.creds), &full);
            let url = action.sign(PRESIGN);
            let req = self
                .http
                .head(url.as_str())
                .map_err(|_| IggyError::CannotReadFileMetadata)?
                .build();
            let resp = self.execute(req, "HEAD").await?;
            if !resp.status().is_success() {
                return Err(IggyError::CannotReadFileMetadata);
            }
            let size = resp
                .headers()
                .get("content-length")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .ok_or(IggyError::CannotReadFileMetadata)?;
            let etag = resp
                .headers()
                .get("etag")
                .and_then(|v| v.to_str().ok())
                .map(strip_etag);
            Ok(ObjectMeta {
                key: full,
                size,
                etag,
            })
        }

        async fn list_prefix(&self, prefix: &str) -> Result<Vec<ObjectMeta>, IggyError> {
            let full_prefix = self.full_key(prefix);
            let mut out = Vec::new();
            let mut continuation: Option<String> = None;
            loop {
                let mut action = self.bucket.list_objects_v2(Some(&self.creds));
                action.with_prefix(&full_prefix);
                if let Some(token) = &continuation {
                    action.with_continuation_token(token);
                }
                let url = action.sign(PRESIGN);
                let req = self
                    .http
                    .get(url.as_str())
                    .map_err(|_| IggyError::CannotReadFile)?
                    .build();
                let resp = self.execute(req, "ListObjectsV2").await?;
                require_2xx(&resp, "ListObjectsV2")?;
                let body = resp.text().await.map_err(|_| IggyError::CannotReadFile)?;
                let parsed =
                    rusty_s3::actions::ListObjectsV2::parse_response(&body).map_err(|e| {
                        tracing::warn!("{COMPONENT} (error: {e}) - parse ListObjectsV2");
                        IggyError::CannotReadFile
                    })?;
                for content in parsed.contents {
                    out.push(ObjectMeta {
                        key: content.key,
                        size: content.size,
                        etag: Some(strip_etag(&content.etag)),
                    });
                }
                continuation = parsed.next_continuation_token;
                if continuation.is_none() {
                    break;
                }
            }
            Ok(out)
        }

        async fn delete(&self, key: &str) -> Result<(), IggyError> {
            let key = self.full_key(key);
            let action = self.bucket.delete_object(Some(&self.creds), &key);
            let url = action.sign(PRESIGN);
            let req = self
                .http
                .delete(url.as_str())
                .map_err(|_| IggyError::CannotDeleteFile)?
                .build();
            let resp = self.execute(req, "DELETE").await?;
            // S3 returns 204 (success) for both existing-and-deleted and missing.
            if resp.status().is_success() {
                Ok(())
            } else {
                Err(IggyError::CannotDeleteFile)
            }
        }
    }

    pub use S3Storage as PubExport;

    struct S3Multipart {
        bucket: Bucket,
        creds: Credentials,
        http: cyper::Client,
        key: String,
        upload_id: String,
        parts: Vec<String>,
        next_part: u16,
    }

    #[async_trait(?Send)]
    impl MultipartHandle for S3Multipart {
        async fn upload_part(&mut self, bytes: Bytes) -> Result<(), IggyError> {
            let action = self.bucket.upload_part(
                Some(&self.creds),
                &self.key,
                self.next_part,
                &self.upload_id,
            );
            let url = action.sign(PRESIGN);
            let req = self
                .http
                .put(url.as_str())
                .map_err(|_| IggyError::CannotWriteToFile)?
                .body(bytes.to_vec())
                .build();
            let resp = self
                .http
                .execute(req)
                .await
                .map_err(|_| IggyError::CannotWriteToFile)?;
            if !resp.status().is_success() {
                return Err(IggyError::CannotWriteToFile);
            }
            // AWS wraps ETag in quotes; rusty-s3's CompleteMultipartUpload
            // re-wraps when serializing the XML body, so we strip here.
            let etag = resp
                .headers()
                .get("etag")
                .and_then(|v| v.to_str().ok())
                .map(strip_etag)
                .ok_or(IggyError::CannotWriteToFile)?;
            self.parts.push(etag);
            self.next_part = self.next_part.saturating_add(1);
            Ok(())
        }

        async fn complete(self: Box<Self>) -> Result<String, IggyError> {
            let action = self.bucket.complete_multipart_upload(
                Some(&self.creds),
                &self.key,
                &self.upload_id,
                self.parts.iter().map(|s| s.as_ref()),
            );
            let url = action.sign(PRESIGN);
            let body = action.body();
            let req = self
                .http
                .post(url.as_str())
                .map_err(|_| IggyError::CannotWriteToFile)?
                .header("Content-Type", "application/xml")
                .map_err(|_| IggyError::CannotWriteToFile)?
                .body(body.into_bytes())
                .build();
            let resp = self
                .http
                .execute(req)
                .await
                .map_err(|_| IggyError::CannotWriteToFile)?;
            if !resp.status().is_success() {
                return Err(IggyError::CannotWriteToFile);
            }
            // S3 sometimes returns 200 with an embedded <Error>; treat that
            // as a failure too.
            let body = resp
                .text()
                .await
                .map_err(|_| IggyError::CannotWriteToFile)?;
            if body.contains("<Error>") {
                return Err(IggyError::CannotWriteToFile);
            }
            Ok(format!("multipart-{}", self.upload_id))
        }

        async fn abort(self: Box<Self>) -> Result<(), IggyError> {
            let action =
                self.bucket
                    .abort_multipart_upload(Some(&self.creds), &self.key, &self.upload_id);
            let url = action.sign(PRESIGN);
            let req = self
                .http
                .delete(url.as_str())
                .map_err(|_| IggyError::CannotDeleteFile)?
                .build();
            let _ = self.http.execute(req).await; // best-effort
            Ok(())
        }
    }
}

#[cfg(feature = "object-storage")]
pub use s3::PubExport as S3Storage;

// =====================================================================
// InMemoryStorage — test-only, HashMap-backed
// =====================================================================

#[cfg(test)]
mod in_memory {
    use super::*;
    use std::collections::{BTreeMap, HashMap};
    use std::sync::{Arc, Mutex};

    /// Test-only `ObjectStorage` backed by a sorted in-memory map.
    /// Replaces what `opendal::services::Memory` would have given us.
    #[derive(Debug, Default, Clone)]
    pub(super) struct InMemoryStorage {
        state: Arc<Mutex<InMemoryState>>,
    }

    #[derive(Debug, Default)]
    struct InMemoryState {
        objects: BTreeMap<String, Bytes>,
        in_progress: HashMap<u64, InProgress>,
        next_upload_id: u64,
        next_etag: u64,
    }

    #[derive(Debug)]
    struct InProgress {
        key: String,
        parts: Vec<Bytes>,
    }

    impl InMemoryStorage {
        pub(super) fn new() -> Self {
            Self::default()
        }

        fn lock(&self) -> std::sync::MutexGuard<'_, InMemoryState> {
            // Test-only; poisoning is fatal anyway.
            self.state.lock().expect("InMemoryStorage state poisoned")
        }
    }

    #[async_trait(?Send)]
    impl ObjectStorage for InMemoryStorage {
        async fn put(&self, key: &str, bytes: Bytes) -> Result<(), IggyError> {
            self.lock().objects.insert(key.to_owned(), bytes);
            Ok(())
        }

        async fn put_if_absent(&self, key: &str, bytes: Bytes) -> Result<(), IggyError> {
            let mut s = self.lock();
            if s.objects.contains_key(key) {
                return Err(IggyError::CannotOverwriteFile);
            }
            s.objects.insert(key.to_owned(), bytes);
            Ok(())
        }

        async fn put_multipart(&self, key: &str) -> Result<Box<dyn MultipartHandle>, IggyError> {
            let mut s = self.lock();
            let upload_id = s.next_upload_id;
            s.next_upload_id += 1;
            s.in_progress.insert(
                upload_id,
                InProgress {
                    key: key.to_owned(),
                    parts: Vec::new(),
                },
            );
            Ok(Box::new(InMemoryMultipart {
                state: self.state.clone(),
                upload_id,
            }))
        }

        async fn get_range(&self, key: &str, range: Range<u64>) -> Result<Bytes, IggyError> {
            let s = self.lock();
            let obj = s.objects.get(key).ok_or(IggyError::CannotReadFile)?;
            let start = range.start as usize;
            let end = (range.end as usize).min(obj.len());
            if start > obj.len() {
                return Err(IggyError::CannotReadFile);
            }
            Ok(obj.slice(start..end))
        }

        async fn head(&self, key: &str) -> Result<ObjectMeta, IggyError> {
            let s = self.lock();
            let obj = s
                .objects
                .get(key)
                .ok_or(IggyError::CannotReadFileMetadata)?;
            Ok(ObjectMeta {
                key: key.to_owned(),
                size: obj.len() as u64,
                etag: None,
            })
        }

        async fn list_prefix(&self, prefix: &str) -> Result<Vec<ObjectMeta>, IggyError> {
            let s = self.lock();
            Ok(s.objects
                .range(prefix.to_owned()..)
                .take_while(|(k, _)| k.starts_with(prefix))
                .map(|(k, v)| ObjectMeta {
                    key: k.clone(),
                    size: v.len() as u64,
                    etag: None,
                })
                .collect())
        }

        async fn delete(&self, key: &str) -> Result<(), IggyError> {
            self.lock().objects.remove(key);
            Ok(())
        }
    }

    struct InMemoryMultipart {
        state: Arc<Mutex<InMemoryState>>,
        upload_id: u64,
    }

    #[async_trait(?Send)]
    impl MultipartHandle for InMemoryMultipart {
        async fn upload_part(&mut self, bytes: Bytes) -> Result<(), IggyError> {
            let mut s = self.state.lock().expect("InMemoryStorage state poisoned");
            let in_progress = s
                .in_progress
                .get_mut(&self.upload_id)
                .ok_or(IggyError::CannotAppendToFile)?;
            in_progress.parts.push(bytes);
            Ok(())
        }

        async fn complete(self: Box<Self>) -> Result<String, IggyError> {
            let mut s = self.state.lock().expect("InMemoryStorage state poisoned");
            let in_progress = s
                .in_progress
                .remove(&self.upload_id)
                .ok_or(IggyError::CannotAppendToFile)?;
            let total: usize = in_progress.parts.iter().map(|b| b.len()).sum();
            let mut buf = bytes::BytesMut::with_capacity(total);
            for part in in_progress.parts {
                buf.extend_from_slice(&part);
            }
            s.objects.insert(in_progress.key, buf.freeze());
            let etag_id = s.next_etag;
            s.next_etag += 1;
            Ok(format!("im-etag-{etag_id}"))
        }

        async fn abort(self: Box<Self>) -> Result<(), IggyError> {
            let mut s = self.state.lock().expect("InMemoryStorage state poisoned");
            s.in_progress.remove(&self.upload_id);
            Ok(())
        }
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::in_memory::InMemoryStorage;
    use super::*;
    use bytes::Bytes;

    fn store() -> InMemoryStorage {
        InMemoryStorage::new()
    }

    #[compio::test]
    async fn put_and_head_round_trip() {
        let s = store();
        s.put("a/b", Bytes::from_static(b"hello")).await.unwrap();
        let meta = s.head("a/b").await.unwrap();
        assert_eq!(meta.size, 5);
    }

    #[compio::test]
    async fn get_range_returns_slice() {
        let s = store();
        s.put("k", Bytes::from_static(b"abcdef")).await.unwrap();
        let got = s.get_range("k", 1..4).await.unwrap();
        assert_eq!(got.as_ref(), b"bcd");
    }

    #[compio::test]
    async fn put_if_absent_first_wins() {
        let s = store();
        s.put_if_absent("k", Bytes::from_static(b"first"))
            .await
            .unwrap();
        let err = s
            .put_if_absent("k", Bytes::from_static(b"second"))
            .await
            .unwrap_err();
        assert!(matches!(err, IggyError::CannotOverwriteFile));
    }

    #[compio::test]
    async fn list_prefix_returns_sorted_matches() {
        let s = store();
        s.put("a/1", Bytes::from_static(b"_")).await.unwrap();
        s.put("a/2", Bytes::from_static(b"_")).await.unwrap();
        s.put("b/1", Bytes::from_static(b"_")).await.unwrap();
        let listed = s.list_prefix("a/").await.unwrap();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].key, "a/1");
        assert_eq!(listed[1].key, "a/2");
    }

    #[compio::test]
    async fn delete_is_idempotent() {
        let s = store();
        s.put("k", Bytes::from_static(b"_")).await.unwrap();
        s.delete("k").await.unwrap();
        s.delete("k").await.unwrap(); // second time also OK
        assert!(s.head("k").await.is_err());
    }

    #[compio::test]
    async fn delete_many_calls_default_path() {
        let s = store();
        s.put("a", Bytes::from_static(b"_")).await.unwrap();
        s.put("b", Bytes::from_static(b"_")).await.unwrap();
        s.delete_many(&["a", "b"]).await.unwrap();
        assert!(s.list_prefix("").await.unwrap().is_empty());
    }

    #[compio::test]
    async fn multipart_complete_assembles_object() {
        let s = store();
        let mut h = s.put_multipart("k").await.unwrap();
        h.upload_part(Bytes::from_static(b"hel")).await.unwrap();
        h.upload_part(Bytes::from_static(b"lo")).await.unwrap();
        let etag = h.complete().await.unwrap();
        assert!(etag.starts_with("im-etag-"));

        let got = s.get_range("k", 0..5).await.unwrap();
        assert_eq!(got.as_ref(), b"hello");
    }

    #[compio::test]
    async fn multipart_abort_does_not_publish() {
        let s = store();
        let mut h = s.put_multipart("k").await.unwrap();
        h.upload_part(Bytes::from_static(b"hello")).await.unwrap();
        h.abort().await.unwrap();
        assert!(s.head("k").await.is_err());
    }

    // ---- CompioFsStorage round-trips ----
    //
    // These use a tempdir and exercise the same trait surface. They run on
    // the compio runtime exactly like the in-memory tests above.

    #[compio::test]
    async fn fs_put_get_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let key = tmp.path().join("obj").to_string_lossy().into_owned();
        let s = CompioFsStorage;
        s.put(&key, Bytes::from_static(b"hello")).await.unwrap();
        let got = s.get_range(&key, 0..5).await.unwrap();
        assert_eq!(got.as_ref(), b"hello");
    }

    #[compio::test]
    async fn fs_put_if_absent_rejects_existing() {
        let tmp = tempfile::tempdir().unwrap();
        let key = tmp.path().join("obj").to_string_lossy().into_owned();
        let s = CompioFsStorage;
        s.put_if_absent(&key, Bytes::from_static(b"a"))
            .await
            .unwrap();
        let err = s
            .put_if_absent(&key, Bytes::from_static(b"b"))
            .await
            .unwrap_err();
        assert!(matches!(err, IggyError::CannotOverwriteFile));
    }

    #[compio::test]
    async fn fs_multipart_writes_in_order() {
        let tmp = tempfile::tempdir().unwrap();
        let key = tmp.path().join("obj").to_string_lossy().into_owned();
        let s = CompioFsStorage;
        let mut h = s.put_multipart(&key).await.unwrap();
        h.upload_part(Bytes::from_static(b"part1-")).await.unwrap();
        h.upload_part(Bytes::from_static(b"part2")).await.unwrap();
        h.complete().await.unwrap();

        let got = s.get_range(&key, 0..11).await.unwrap();
        assert_eq!(got.as_ref(), b"part1-part2");
    }

    // ---- BufferedMultipartWriter ----
    //
    // The buffer is what makes typical sub-MiB iggy flushes legal under
    // S3's 5 MiB-per-part minimum. These tests use InMemoryStorage and
    // exercise the four states from the Phase 0 spec: below threshold
    // (no parts), crossing threshold (one part), seal-with-residual
    // (final part flushed), and small-segment-fallback (multipart aborted,
    // single PUT).

    fn rc_store() -> Rc<dyn ObjectStorage> {
        Rc::new(InMemoryStorage::new())
    }

    #[compio::test]
    async fn buffered_writer_below_threshold_no_parts() {
        let s = rc_store();
        let mut w = BufferedMultipartWriter::open(s.clone(), "k", S3_MIN_PART_SIZE)
            .await
            .unwrap();
        w.append(&[0u8; 1024]).await.unwrap();
        assert_eq!(w.parts_uploaded(), 0);
        assert_eq!(w.buffered_bytes(), 1024);
    }

    #[compio::test]
    async fn buffered_writer_crossing_threshold_emits_one_part() {
        let s = rc_store();
        let mut w = BufferedMultipartWriter::open(s.clone(), "k", S3_MIN_PART_SIZE)
            .await
            .unwrap();
        // 5 MiB exactly → one part flushed, buffer empty.
        w.append(&vec![0u8; S3_MIN_PART_SIZE]).await.unwrap();
        assert_eq!(w.parts_uploaded(), 1);
        assert_eq!(w.buffered_bytes(), 0);
    }

    #[compio::test]
    async fn buffered_writer_seal_with_residual_completes() {
        let s = rc_store();
        let mut w = BufferedMultipartWriter::open(s.clone(), "k", S3_MIN_PART_SIZE)
            .await
            .unwrap();
        // 5 MiB + 1 KiB residual → 1 part + 1 final small part.
        w.append(&vec![0u8; S3_MIN_PART_SIZE]).await.unwrap();
        w.append(&vec![1u8; 1024]).await.unwrap();
        assert_eq!(w.parts_uploaded(), 1);
        assert_eq!(w.buffered_bytes(), 1024);
        let etag = w.seal().await.unwrap();
        assert!(etag.starts_with("im-etag-"));

        let meta = s.head("k").await.unwrap();
        assert_eq!(meta.size as usize, S3_MIN_PART_SIZE + 1024);
    }

    #[compio::test]
    async fn buffered_writer_small_segment_fallback_does_single_put() {
        let s = rc_store();
        let mut w = BufferedMultipartWriter::open(s.clone(), "k", S3_MIN_PART_SIZE)
            .await
            .unwrap();
        // 100 bytes — below S3 minimum, no parts yet → fallback path.
        w.append(b"hello world").await.unwrap();
        assert_eq!(w.parts_uploaded(), 0);
        let etag = w.seal().await.unwrap();
        // Fallback path returns empty etag (no multipart final etag).
        assert!(etag.is_empty());

        let got = s.get_range("k", 0..11).await.unwrap();
        assert_eq!(got.as_ref(), b"hello world");
    }

    // ---- S3 wire test (gated on IGGY_TEST_MINIO) ----
    //
    // Skips by default. With MinIO running on localhost:9000 (or a custom
    // S3-compatible endpoint), set the env to exercise the real wire code.

    #[cfg(feature = "object-storage")]
    #[compio::test]
    async fn s3_minio_round_trip() {
        if std::env::var("IGGY_TEST_MINIO").is_err() {
            return;
        }
        use crate::configs::system::ObjectStorageConfig;
        let _ = rustls::crypto::ring::default_provider().install_default();
        let config = ObjectStorageConfig {
            service: "s3".into(),
            bucket: std::env::var("IGGY_TEST_MINIO_BUCKET").unwrap_or_else(|_| "iggy-test".into()),
            region: std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".into()),
            endpoint: std::env::var("S3_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:9000".into()),
            prefix: format!(
                "test-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_micros()
            ),
            multipart_part_size: iggy_common::IggyByteSize::from(8 * 1024 * 1024_u64),
            ack_after_upload: true,
            access_key_id: std::env::var("AWS_ACCESS_KEY_ID")
                .unwrap_or_else(|_| "minioadmin".into()),
            secret_access_key: std::env::var("AWS_SECRET_ACCESS_KEY")
                .unwrap_or_else(|_| "minioadmin".into()),
            profile: String::new(),
        };
        let s3 = S3Storage::from_config(&config).expect("S3Storage::from_config");

        // PUT + GET round-trip on a small object.
        s3.put("hello.bin", Bytes::from_static(b"hello"))
            .await
            .expect("put");
        let got = s3.get_range("hello.bin", 0..5).await.expect("get_range");
        assert_eq!(got.as_ref(), b"hello");
        s3.delete("hello.bin").await.expect("delete");
    }
}
