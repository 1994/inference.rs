//! Error HTTP adapter.
use axum::{Json, http::StatusCode, response::IntoResponse, response::Response};
use infer_core::{Error, ErrorCode};

pub(super) type HttpResult<T> = Result<T, HttpError>;
pub struct HttpError(Error, Option<StatusCode>);
impl From<Error> for HttpError {
    fn from(error: Error) -> Self {
        Self(error, None)
    }
}
impl IntoResponse for HttpError {
    fn into_response(self) -> Response {
        let status = match self.0.code {
            ErrorCode::InvalidInput => StatusCode::BAD_REQUEST,
            ErrorCode::Unsupported => StatusCode::NOT_IMPLEMENTED,
            ErrorCode::Capacity => StatusCode::TOO_MANY_REQUESTS,
            ErrorCode::NotFound => StatusCode::NOT_FOUND,
            ErrorCode::Conflict => StatusCode::CONFLICT,
            ErrorCode::Invariant => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorCode::Backend => StatusCode::SERVICE_UNAVAILABLE,
        };
        (self.1.unwrap_or(status), Json(self.0)).into_response()
    }
}
impl From<axum::extract::rejection::JsonRejection> for HttpError {
    fn from(rejection: axum::extract::rejection::JsonRejection) -> Self {
        Self(
            Error::invalid(rejection.body_text()),
            Some(rejection.status()),
        )
    }
}
