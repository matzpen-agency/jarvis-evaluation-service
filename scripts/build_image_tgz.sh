#!/usr/bin/env bash
# build_image_tgz.sh
# Builds the evaluation service image for linux/amd64 (OpenShift) and saves it
# as a gzipped tarball, ready to copy into the private network and load there.
#
# Usage:  scripts/build_image_tgz.sh [tag]        (default tag: the git short SHA)
# Output: dist/jarvis-evaluation-service-<tag>.tgz (+ .sha256)
#
# The build installs a private git dependency over SSH, so an ssh-agent holding
# a GitHub key with access to matzpen-agency/python-core-utils must be running.

set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE_NAME="${IMAGE_NAME:-jarvis-evaluation-service}"
TAG="${1:-$(git rev-parse --short HEAD)}"
PLATFORM="${PLATFORM:-linux/amd64}"
OUT_DIR="dist"
OUT_FILE="${OUT_DIR}/${IMAGE_NAME}-${TAG}.tgz"

if ! ssh-add -l >/dev/null 2>&1; then
  echo "❌ No SSH key in ssh-agent. The build needs one to fetch python-core-utils:"
  echo "   ssh-add ~/.ssh/<your-github-key>"
  exit 1
fi

if [ -n "$(git status --porcelain -- src pyproject.toml uv.lock Dockerfile)" ]; then
  echo "⚠️  Uncommitted changes in src/, pyproject.toml, uv.lock or Dockerfile are included in this image."
fi

echo "🔨 Building ${IMAGE_NAME}:${TAG} (${PLATFORM})..."
docker buildx build \
  --platform "${PLATFORM}" \
  --ssh default \
  --load \
  -t "${IMAGE_NAME}:${TAG}" \
  -t "${IMAGE_NAME}:latest" \
  .

mkdir -p "${OUT_DIR}"
echo "📦 Saving ${IMAGE_NAME}:${TAG} to ${OUT_FILE}..."
docker save "${IMAGE_NAME}:${TAG}" "${IMAGE_NAME}:latest" | gzip > "${OUT_FILE}"

(cd "${OUT_DIR}" && shasum -a 256 "$(basename "${OUT_FILE}")" > "$(basename "${OUT_FILE}").sha256")

echo "✅ Done: ${OUT_FILE} ($(du -h "${OUT_FILE}" | cut -f1))"
echo "   SHA-256: $(cut -d' ' -f1 "${OUT_FILE}.sha256")"
echo ""
echo "In the private network:"
echo "  sha256sum -c $(basename "${OUT_FILE}").sha256      # check the copy is intact"
echo "  docker load -i $(basename "${OUT_FILE}")            # or: podman load -i ..."
echo "  docker tag ${IMAGE_NAME}:${TAG} <registry>/<project>/${IMAGE_NAME}:${TAG}"
echo "  docker push <registry>/<project>/${IMAGE_NAME}:${TAG}"
echo ""
echo "Runtime config is not in the image — set it as environment variables (see .env.example):"
echo "  LANGFUSE_*, JEEN_*, the agent URL, LLM_* or OPENAI_API_KEY (order judge),"
echo "  AGENT_TRINO_* (+ mount the Trino client certs at /app/certs),"
echo "  and mount a volume at /app/data to keep order-judge decisions."
