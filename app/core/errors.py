from enum import StrEnum


class AgentErrorCode(StrEnum):
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_RATE_LIMITED = "provider_rate_limited"
    PROVIDER_QUOTA_EXHAUSTED = "provider_quota_exhausted"
    PROVIDER_AUTHENTICATION_FAILED = "provider_authentication_failed"
    PROVIDER_REQUEST_REJECTED = "provider_request_rejected"
    TRANSIENT_PROVIDER_FAILURE = "transient_provider_failure"
    PROVIDER_FAILURE = "provider_failure"
    MALFORMED_PROVIDER_OUTPUT = "malformed_provider_output"
    INVALID_PROVIDER_OUTPUT = "invalid_provider_output"
    ILLEGAL_TOOL = "illegal_tool"
    INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
    UNKNOWN_REQUESTED_BRAND = "unknown_requested_brand"
    TURN_LIMIT_EXHAUSTED = "turn_limit_exhausted"
