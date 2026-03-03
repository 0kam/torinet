#!/usr/bin/env bash
# ============================================================
# ToriNet Step 02: 音声ダウンロード一括実行スクリプト
#
# 使い方:
#   # フルダウンロード（XC + iNat S3 + iNat API + Macaulay）
#   bash steps/02_data_collection/download_all.sh
#
#   # XCのみ
#   bash steps/02_data_collection/download_all.sh --xc-only
#
#   # iNatのみ（S3 + API両方）
#   bash steps/02_data_collection/download_all.sh --inat-only
#
#   # iNat APIのみ（メタデータ収集 + DL）
#   bash steps/02_data_collection/download_all.sh --inat-api-only
#
#   # Macaulay Libraryのみ
#   bash steps/02_data_collection/download_all.sh --ml-only
#
#   # tmux/screenで実行推奨（長時間）
#   tmux new -s torinet-dl 'bash steps/02_data_collection/download_all.sh'
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
LOG_DIR="$PROJECT_ROOT/steps/02_data_collection/logs"
XC_SCRIPT="$SCRIPT_DIR/collect_xc.py"
INAT_SCRIPT="$SCRIPT_DIR/collect_inat.py"
INAT_API_SCRIPT="$SCRIPT_DIR/collect_inat_api.py"
ML_SCRIPT="$SCRIPT_DIR/collect_macaulay.py"

# ── 設定 ──
XC_FORMAT="mp3"           # mp3 or wav
XC_MAX_RETRIES=10         # プロセスクラッシュ時の最大リトライ
INAT_MAX_RETRIES=5
INAT_API_MAX_RETRIES=10   # iNat APIは帯域制限で中断しやすい
ML_MAX_RETRIES=10         # Macaulay Library
RETRY_WAIT_SEC=30         # リトライ間の待機秒数

# ── 引数解析 ──
RUN_XC=true
RUN_INAT=true
RUN_INAT_API=true
RUN_ML=true
for arg in "$@"; do
    case "$arg" in
        --xc-only)       RUN_INAT=false; RUN_INAT_API=false; RUN_ML=false ;;
        --inat-only)     RUN_XC=false; RUN_ML=false ;;
        --inat-api-only) RUN_XC=false; RUN_INAT=false; RUN_ML=false ;;
        --ml-only)       RUN_XC=false; RUN_INAT=false; RUN_INAT_API=false ;;
        --wav)           XC_FORMAT="wav" ;;
        --help|-h)
            echo "Usage: $0 [--xc-only] [--inat-only] [--inat-api-only] [--ml-only] [--wav]"
            echo ""
            echo "Options:"
            echo "  --xc-only        Xeno-cantoのみダウンロード"
            echo "  --inat-only      iNat全体（S3 + API）のみダウンロード"
            echo "  --inat-api-only  iNat APIのみ（メタデータ収集+DL）"
            echo "  --ml-only        Macaulay Libraryのみ（メタデータ収集+DL）"
            echo "  --wav            XCをWAV変換して保存（デフォルト: MP3保持）"
            exit 0
            ;;
    esac
done

# ── ユーティリティ ──
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

elapsed() {
    local start=$1
    local now
    now=$(date +%s)
    local diff=$((now - start))
    printf '%dh%02dm%02ds' $((diff/3600)) $((diff%3600/60)) $((diff%60))
}

# ── 前提チェック ──
log "=== ToriNet Audio Download ==="
log "Project: $PROJECT_ROOT"
log "Format: $XC_FORMAT"

if [ ! -f "$VENV_PYTHON" ]; then
    log "ERROR: Python venv not found at $VENV_PYTHON"
    exit 1
fi

if [ "$XC_FORMAT" = "wav" ]; then
    if ! command -v ffmpeg &>/dev/null; then
        log "ERROR: ffmpeg required for WAV conversion. Install or use MP3."
        exit 1
    fi
fi

if $RUN_INAT; then
    VENV_AWS="$PROJECT_ROOT/.venv/bin/aws"
    if [ ! -f "$VENV_AWS" ] && ! command -v aws &>/dev/null; then
        log "ERROR: aws cli required for iNat S3 download."
        exit 1
    fi
fi

# ============================================================
# Xeno-canto ダウンロード
# ============================================================
if $RUN_XC; then
    XC_LOG="$LOG_DIR/xc_download_${TIMESTAMP}.log"
    log ""
    log "=== Xeno-canto Download ==="
    log "Log: $XC_LOG"

    xc_start=$(date +%s)
    xc_attempt=0
    xc_done=false

    while [ $xc_attempt -lt $XC_MAX_RETRIES ] && ! $xc_done; do
        xc_attempt=$((xc_attempt + 1))
        log "Attempt $xc_attempt/$XC_MAX_RETRIES ..."

        set +e
        "$VENV_PYTHON" "$XC_SCRIPT" --download --format "$XC_FORMAT" \
            2>&1 | tee -a "$XC_LOG"
        xc_exit=$?
        set -e

        if [ $xc_exit -eq 0 ]; then
            xc_done=true
            log "XC download completed successfully ($(elapsed $xc_start))"
        else
            log "XC download exited with code $xc_exit"

            # 進捗確認: DL済みファイル数をチェック
            progress_file="$HOME/NAS/nasbi/ToriNET/audio/xeno-canto/metadata/download_progress.json"
            if [ -f "$progress_file" ]; then
                tracked=$("$VENV_PYTHON" -c "
import json
with open('$progress_file') as f:
    print(len(json.load(f)))
" 2>/dev/null || echo "?")
                log "  Progress so far: $tracked files tracked"
            fi

            if [ $xc_attempt -lt $XC_MAX_RETRIES ]; then
                log "  Retrying in ${RETRY_WAIT_SEC}s (files already downloaded will be skipped)..."
                sleep "$RETRY_WAIT_SEC"
            else
                log "  Max retries reached. Run this script again to continue."
            fi
        fi
    done

    log "XC total time: $(elapsed $xc_start)"
    echo ""
fi

# ============================================================
# iNat Sounds 2024 ダウンロード
# ============================================================
if $RUN_INAT; then
    INAT_LOG="$LOG_DIR/inat_download_${TIMESTAMP}.log"
    log ""
    log "=== iNat Sounds 2024 Download ==="
    log "Log: $INAT_LOG"

    inat_start=$(date +%s)
    inat_attempt=0
    inat_done=false

    while [ $inat_attempt -lt $INAT_MAX_RETRIES ] && ! $inat_done; do
        inat_attempt=$((inat_attempt + 1))
        log "Attempt $inat_attempt/$INAT_MAX_RETRIES ..."

        set +e
        "$VENV_PYTHON" "$INAT_SCRIPT" --download \
            2>&1 | tee -a "$INAT_LOG"
        inat_exit=$?
        set -e

        if [ $inat_exit -eq 0 ]; then
            inat_done=true
            log "iNat download completed successfully ($(elapsed $inat_start))"
        else
            log "iNat download exited with code $inat_exit"

            # 進捗確認: filtered内のファイル数
            inat_filtered="$HOME/NAS/nasbi/ToriNET/audio/inat-sounds/filtered"
            if [ -d "$inat_filtered" ]; then
                f_count=$(find "$inat_filtered" -type f 2>/dev/null | wc -l)
                log "  Progress so far: $f_count files in filtered/"
            fi

            if [ $inat_attempt -lt $INAT_MAX_RETRIES ]; then
                log "  Retrying in ${RETRY_WAIT_SEC}s..."
                sleep "$RETRY_WAIT_SEC"
            else
                log "  Max retries reached."
            fi
        fi
    done

    log "iNat S3 total time: $(elapsed $inat_start)"
    echo ""
fi

# ============================================================
# iNat API 追加音声（メタデータ収集 + ダウンロード）
# ============================================================
if $RUN_INAT_API; then
    INAT_API_LOG="$LOG_DIR/inat_api_download_${TIMESTAMP}.log"
    log ""
    log "=== iNat API Additional Audio ==="
    log "Log: $INAT_API_LOG"

    # Phase 1: メタデータ収集
    inat_api_start=$(date +%s)
    inat_api_attempt=0
    inat_api_meta_done=false

    log "--- Phase 1: Metadata Collection ---"
    while [ $inat_api_attempt -lt $INAT_API_MAX_RETRIES ] && ! $inat_api_meta_done; do
        inat_api_attempt=$((inat_api_attempt + 1))
        log "Metadata attempt $inat_api_attempt/$INAT_API_MAX_RETRIES ..."

        set +e
        "$VENV_PYTHON" "$INAT_API_SCRIPT" --metadata-only \
            2>&1 | tee -a "$INAT_API_LOG"
        inat_api_exit=$?
        set -e

        if [ $inat_api_exit -eq 0 ]; then
            inat_api_meta_done=true
            log "iNat API metadata completed ($(elapsed $inat_api_start))"
        else
            log "iNat API metadata exited with code $inat_api_exit"

            progress_file="$HOME/NAS/nasbi/ToriNET/audio/inat-api/metadata/collection_progress.json"
            if [ -f "$progress_file" ]; then
                tracked=$("$VENV_PYTHON" -c "
import json
with open('$progress_file') as f:
    d = json.load(f)
print(len(d.get('completed_species', [])))
" 2>/dev/null || echo "?")
                log "  Progress: $tracked species collected"
            fi

            if [ $inat_api_attempt -lt $INAT_API_MAX_RETRIES ]; then
                log "  Retrying in ${RETRY_WAIT_SEC}s..."
                sleep "$RETRY_WAIT_SEC"
            else
                log "  Max retries for metadata. Skipping download phase."
            fi
        fi
    done

    # Phase 2: ダウンロード（メタデータ完了後のみ）
    if $inat_api_meta_done; then
        log ""
        log "--- Phase 2: Audio Download ---"
        inat_api_dl_attempt=0
        inat_api_dl_done=false

        while [ $inat_api_dl_attempt -lt $INAT_API_MAX_RETRIES ] && ! $inat_api_dl_done; do
            inat_api_dl_attempt=$((inat_api_dl_attempt + 1))
            log "Download attempt $inat_api_dl_attempt/$INAT_API_MAX_RETRIES ..."

            set +e
            "$VENV_PYTHON" "$INAT_API_SCRIPT" --download \
                2>&1 | tee -a "$INAT_API_LOG"
            inat_api_dl_exit=$?
            set -e

            if [ $inat_api_dl_exit -eq 0 ]; then
                inat_api_dl_done=true
                log "iNat API download completed ($(elapsed $inat_api_start))"
            else
                log "iNat API download exited with code $inat_api_dl_exit"

                dl_progress="$HOME/NAS/nasbi/ToriNET/audio/inat-api/metadata/download_progress.json"
                if [ -f "$dl_progress" ]; then
                    tracked=$("$VENV_PYTHON" -c "
import json
with open('$dl_progress') as f:
    d = json.load(f)
print(len(d.get('files', {})))
" 2>/dev/null || echo "?")
                    log "  Progress: $tracked files tracked"
                fi

                if [ $inat_api_dl_attempt -lt $INAT_API_MAX_RETRIES ]; then
                    log "  Retrying in ${RETRY_WAIT_SEC}s..."
                    sleep "$RETRY_WAIT_SEC"
                else
                    log "  Max retries reached for download."
                fi
            fi
        done
    fi

    log "iNat API total time: $(elapsed $inat_api_start)"
    echo ""
fi

# ============================================================
# Macaulay Library ダウンロード
# ============================================================
if $RUN_ML; then
    ML_LOG="$LOG_DIR/ml_download_${TIMESTAMP}.log"
    log ""
    log "=== Macaulay Library Download ==="
    log "Log: $ML_LOG"

    # Phase 1: メタデータ収集
    ml_start=$(date +%s)
    ml_attempt=0
    ml_meta_done=false

    log "--- Phase 1: Metadata Collection ---"
    while [ $ml_attempt -lt $ML_MAX_RETRIES ] && ! $ml_meta_done; do
        ml_attempt=$((ml_attempt + 1))
        log "Metadata attempt $ml_attempt/$ML_MAX_RETRIES ..."

        set +e
        "$VENV_PYTHON" "$ML_SCRIPT" --metadata-only \
            2>&1 | tee -a "$ML_LOG"
        ml_exit=$?
        set -e

        if [ $ml_exit -eq 0 ]; then
            ml_meta_done=true
            log "ML metadata completed ($(elapsed $ml_start))"
        else
            log "ML metadata exited with code $ml_exit"

            progress_file="$HOME/NAS/nasbi/ToriNET/audio/macaulay/metadata/collection_progress.json"
            if [ -f "$progress_file" ]; then
                tracked=$("$VENV_PYTHON" -c "
import json
with open('$progress_file') as f:
    d = json.load(f)
print(len(d.get('completed_species', [])))
" 2>/dev/null || echo "?")
                log "  Progress: $tracked species collected"
            fi

            if [ $ml_attempt -lt $ML_MAX_RETRIES ]; then
                log "  Retrying in ${RETRY_WAIT_SEC}s..."
                sleep "$RETRY_WAIT_SEC"
            else
                log "  Max retries for metadata. Skipping download phase."
            fi
        fi
    done

    # Phase 2: ダウンロード（メタデータ完了後のみ）
    if $ml_meta_done; then
        log ""
        log "--- Phase 2: Audio Download ---"
        ml_dl_attempt=0
        ml_dl_done=false

        while [ $ml_dl_attempt -lt $ML_MAX_RETRIES ] && ! $ml_dl_done; do
            ml_dl_attempt=$((ml_dl_attempt + 1))
            log "Download attempt $ml_dl_attempt/$ML_MAX_RETRIES ..."

            set +e
            "$VENV_PYTHON" "$ML_SCRIPT" --download \
                2>&1 | tee -a "$ML_LOG"
            ml_dl_exit=$?
            set -e

            if [ $ml_dl_exit -eq 0 ]; then
                ml_dl_done=true
                log "ML download completed ($(elapsed $ml_start))"
            else
                log "ML download exited with code $ml_dl_exit"

                dl_progress="$HOME/NAS/nasbi/ToriNET/audio/macaulay/metadata/download_progress.json"
                if [ -f "$dl_progress" ]; then
                    tracked=$("$VENV_PYTHON" -c "
import json
with open('$dl_progress') as f:
    d = json.load(f)
print(len(d.get('files', {})))
" 2>/dev/null || echo "?")
                    log "  Progress: $tracked files tracked"
                fi

                if [ $ml_dl_attempt -lt $ML_MAX_RETRIES ]; then
                    log "  Retrying in ${RETRY_WAIT_SEC}s..."
                    sleep "$RETRY_WAIT_SEC"
                else
                    log "  Max retries reached for download."
                fi
            fi
        done
    fi

    log "ML total time: $(elapsed $ml_start)"
    echo ""
fi

# ============================================================
# 完了レポート
# ============================================================
log ""
log "============================================================"
log "  Download Complete"
log "============================================================"

if $RUN_XC; then
    xc_dir="$HOME/NAS/nasbi/ToriNET/audio/xeno-canto/wav"
    if [ -d "$xc_dir" ]; then
        xc_species=$(ls -d "$xc_dir"/*/ 2>/dev/null | wc -l)
        xc_files=$(find "$xc_dir" -type f | wc -l)
        xc_size=$(du -sh "$xc_dir" 2>/dev/null | cut -f1)
        log "XC: $xc_files files, $xc_species species, $xc_size"
    fi
fi

if $RUN_INAT; then
    inat_dir="$HOME/NAS/nasbi/ToriNET/audio/inat-sounds/filtered"
    if [ -d "$inat_dir" ]; then
        inat_species=$(ls -d "$inat_dir"/*/ 2>/dev/null | wc -l)
        inat_files=$(find "$inat_dir" -type f | wc -l)
        inat_size=$(du -sh "$inat_dir" 2>/dev/null | cut -f1)
        log "iNat S3:  $inat_files files, $inat_species species, $inat_size"
    fi
fi

if $RUN_INAT_API; then
    inat_api_dir="$HOME/NAS/nasbi/ToriNET/audio/inat-api/audio"
    if [ -d "$inat_api_dir" ]; then
        inat_api_species=$(ls -d "$inat_api_dir"/*/ 2>/dev/null | wc -l)
        inat_api_files=$(find "$inat_api_dir" -type f | wc -l)
        inat_api_size=$(du -sh "$inat_api_dir" 2>/dev/null | cut -f1)
        log "iNat API: $inat_api_files files, $inat_api_species species, $inat_api_size"
    fi
fi

if $RUN_ML; then
    ml_dir="$HOME/NAS/nasbi/ToriNET/audio/macaulay/audio"
    if [ -d "$ml_dir" ]; then
        ml_species=$(ls -d "$ml_dir"/*/ 2>/dev/null | wc -l)
        ml_files=$(find "$ml_dir" -type f | wc -l)
        ml_size=$(du -sh "$ml_dir" 2>/dev/null | cut -f1)
        log "ML:       $ml_files files, $ml_species species, $ml_size"
    fi
fi

log "Logs: $LOG_DIR"
log "Done."
