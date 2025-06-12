SHELL := $(shell which sh)
GOOS := $(shell go env GOOS)
GOARCH := $(shell go env GOARCH)
GOPATH := $(shell go env GOPATH)
GOLANG_CI_LINT := bin/golangci-lint

AGENTRUNTIME_BIN := bin/agentruntime
AGENTRUNTIME_BIN_FILES := bin/agentruntime-linux-amd64 bin/agentruntime-linux-arm64 bin/agentruntime-darwin-amd64 bin/agentruntime-darwin-arm64 bin/agentruntime-windows-amd64.exe

.PHONY: build
build: $(AGENTRUNTIME_BIN)

$(GOLANG_CI_LINT):
	@curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s v1.64.7
	@chmod +x $(GOLANG_CI_LINT)
	@echo "golangci-lint installed"

.PHONY: lint
lint: $(GOLANG_CI_LINT)
	$(GOLANG_CI_LINT) run --timeout 0

.PHONY: test
test:
	go install github.com/joho/godotenv/cmd/godotenv@latest
	CI=true godotenv -f .env.test go test -timeout 15m -p 1 ./...

.PHONY: clean
clean:
	rm -rf bin/*
	rm -f $(GOLANG_CI_LINT)
	@echo "cleared"

.PHONY: agentruntime-windows-*.exe
bin/agentruntime-windows-%.exe:
	GOOS=windows GOARCH=$* go build -o $@ ./cmd/agentruntime

.PHONY: bin/agentruntime-*
bin/agentruntime-%:
	$(eval OS_NAME := $(word 1,$(subst -, ,$*)))
	$(eval ARCH_NAME := $(word 2,$(subst -, ,$*)))
	GOOS=$(OS_NAME) GOARCH=$(ARCH_NAME) go build -o $@ ./cmd/agentruntime

.PHONY: $(AGENTRUNTIME_BIN)
$(AGENTRUNTIME_BIN): bin/agentruntime-$(GOOS)-$(GOARCH)
	ln -sf agentruntime-$(GOOS)-$(GOARCH) $(AGENTRUNTIME_BIN)

.PHONY: install
install:
	go install ./cmd/agentruntime

.PHONY: release
release:
	$(eval NEXT_VERSION := $(shell convco version --bump))
	git tag -a v$(NEXT_VERSION) -m "chore(release): v$(NEXT_VERSION)"
	git push origin v$(NEXT_VERSION)
	convco changelog --max-versions 1 > CHANGELOG.md
	gh release create v$(NEXT_VERSION) --title "v$(NEXT_VERSION)" --notes-file CHANGELOG.md

.PHONY: build-docker-agentruntime
build-docker-agentruntime:
	docker build --push -t ghcr.io/habiliai/agentruntime:latest -f cmd/agentruntime/Dockerfile .
