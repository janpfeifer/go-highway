# Dockerfile for building and testing go-highway
#
# Build: docker build -t go-highway .
# Test:  docker run --rm go-highway

FROM golang:1.23 AS builder

# Install Go 1.26rc1 for SIMD support
RUN go install golang.org/dl/go1.26rc1@latest && \
    go1.26rc1 download

WORKDIR /app

# Copy go.mod and go.sum first for better caching
COPY go.mod go.sum ./
RUN go1.26rc1 mod download

# Copy source code
COPY . .

# Build hwygen
RUN GOEXPERIMENT=simd go1.26rc1 build -o bin/hwygen ./cmd/hwygen

# Run go generate on examples
RUN PATH="/app/bin:$PATH" GOEXPERIMENT=simd go1.26rc1 generate ./examples/...

# Build all packages
RUN GOEXPERIMENT=simd go1.26rc1 build ./...

# Run tests
FROM builder AS tester

# Run all tests
RUN GOEXPERIMENT=simd go1.26rc1 test ./... -v

# Run tests with fallback (HWY_NO_SIMD)
RUN HWY_NO_SIMD=1 GOEXPERIMENT=simd go1.26rc1 test ./... -v

# Final stage - just verify build succeeded
FROM builder AS final

# Default command runs tests
CMD ["sh", "-c", "GOEXPERIMENT=simd go1.26rc1 test ./..."]
