class AppError(Exception):
    """Friendly, user-facing error. Never carries a stack trace to the client."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status
