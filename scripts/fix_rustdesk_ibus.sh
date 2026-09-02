#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly RUN_USER="$(id -un)"
readonly RUN_UID="$(id -u)"
readonly RUN_HOME="$(getent passwd "$RUN_UID" | cut -d: -f6)"
readonly RUNTIME_DIR="/run/user/$RUN_UID"
readonly DBUS_ADDRESS="unix:path=$RUNTIME_DIR/bus"
readonly USER_ENV_FILE="$RUN_HOME/.config/environment.d/20-ibus.conf"
readonly USER_WRAPPER="$RUN_HOME/.local/bin/rustdesk-ibus"
readonly SYSTEM_DROPIN="/etc/systemd/system/rustdesk.service.d/99-rustdesk-ibus.conf"

user_tmp=""
dropin_tmp=""

cleanup() {
    if [ -n "$user_tmp" ]; then
        rm -f -- "$user_tmp"
    fi
    if [ -n "$dropin_tmp" ]; then
        rm -f -- "$dropin_tmp"
    fi
}

trap cleanup EXIT

die() {
    printf '错误：%s\n' "$*" >&2
    exit 1
}

need_command() {
    command -v "$1" >/dev/null 2>&1 || die "缺少命令：$1"
}

write_user_file_if_changed() {
    local target="$1"
    local mode="$2"
    local content="$3"
    local backup

    user_tmp="$(mktemp "$RUN_HOME/.rustdesk-ibus.XXXXXX")"
    printf '%s\n' "$content" >"$user_tmp"

    if [ -e "$target" ] && cmp -s "$user_tmp" "$target"; then
        rm -f -- "$user_tmp"
        user_tmp=""
        return
    fi

    if [ -e "$target" ]; then
        backup="$target.bak.$(date +%Y%m%d%H%M%S)"
        cp -p -- "$target" "$backup"
        printf '已备份原文件：%s\n' "$backup"
    fi

    install -m "$mode" "$user_tmp" "$target"
    rm -f -- "$user_tmp"
    user_tmp=""
}

find_rustdesk_cm_pid() {
    ps -eo pid=,user=,args= | awk -v run_user="$RUN_USER" '
        $2 == run_user && index($0, "rustdesk") && index($0, "--cm") {
            print $1
            exit
        }
    '
}

if [ "$RUN_UID" -eq 0 ]; then
    die "请使用当前桌面用户执行，不要用 root 执行：./$SCRIPT_NAME"
fi

for command_name in awk basename cmp cp cut date getent grep gsettings ibus install mkdir mktemp ps rm seq sleep sudo systemctl tr; do
    need_command "$command_name"
done

[ -d "$RUN_HOME" ] || die "无法确定当前用户家目录：$RUN_HOME"
[ -S "$RUNTIME_DIR/bus" ] || die "当前没有可用的用户 DBus：$RUNTIME_DIR/bus"
systemctl cat rustdesk.service >/dev/null 2>&1 || die "找不到 rustdesk.service"

export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-$DBUS_ADDRESS}"

printf '这个脚本将完成以下操作：\n'
printf '  1. 将本机 IBus 的切换键设为 Ctrl+Space。\n'
printf '  2. 写入 RustDesk/IBus 用户环境。\n'
printf '  3. 写入 RustDesk systemd drop-in。\n'
printf '  4. 重启 rustdesk.service；当前远程连接会短暂断开。\n'
printf '\n确认继续请输入 APPLY： '
read -r confirmation
[ "$confirmation" = "APPLY" ] || {
    printf '已取消，未修改系统。\n'
    exit 0
}

sudo -v || die "sudo 认证失败；没有执行系统服务修改"

mkdir -p -- "$RUN_HOME/.config/environment.d" "$RUN_HOME/.local/bin"

write_user_file_if_changed "$USER_ENV_FILE" 0644 $'GTK_IM_MODULE=ibus\nQT_IM_MODULE=ibus\nXMODIFIERS=@im=ibus'
write_user_file_if_changed "$USER_WRAPPER" 0755 $'#!/bin/sh\nset -eu\n\nexport GTK_IM_MODULE=ibus\nexport QT_IM_MODULE=ibus\nexport XMODIFIERS=@im=ibus\n\nexec /usr/share/rustdesk/rustdesk "$@"'

gsettings set org.freedesktop.ibus.general.hotkey next-engine "['Control+space']"
gsettings set org.freedesktop.ibus.general.hotkey next-engine-in-menu "['Control+space']"
gsettings set org.freedesktop.ibus.general.hotkey trigger "@as []"
gsettings set org.freedesktop.ibus.general preload-engines "['libpinyin', 'xkb:us::eng']"
gsettings set org.freedesktop.ibus.general engines-order "['libpinyin', 'xkb:us::eng']"
ibus engine libpinyin

dropin_tmp="$(mktemp)"
printf '%s\n' \
    '[Service]' \
    'Environment="GTK_IM_MODULE=ibus"' \
    'Environment="QT_IM_MODULE=ibus"' \
    'Environment="XMODIFIERS=@im=ibus"' \
    "Environment=\"DBUS_SESSION_BUS_ADDRESS=$DBUS_ADDRESS\"" \
    >"$dropin_tmp"

sudo install -d -m 0755 /etc/systemd/system/rustdesk.service.d
if sudo test -e "$SYSTEM_DROPIN" && ! sudo cmp -s "$dropin_tmp" "$SYSTEM_DROPIN"; then
    sudo cp -p -- "$SYSTEM_DROPIN" "$SYSTEM_DROPIN.bak.$(date +%Y%m%d%H%M%S)"
    printf '已备份原 systemd drop-in。\n'
fi
sudo install -m 0644 "$dropin_tmp" "$SYSTEM_DROPIN"

sudo systemctl daemon-reload
sudo systemctl restart rustdesk.service
sudo systemctl is-active --quiet rustdesk.service || {
    sudo systemctl --no-pager --full status rustdesk.service || true
    die "rustdesk.service 重启失败"
}

cm_pid=""
for _ in $(seq 1 20); do
    cm_pid="$(find_rustdesk_cm_pid || true)"
    [ -n "$cm_pid" ] && break
    sleep 0.5
done
[ -n "$cm_pid" ] || die "未找到 RustDesk --cm 进程，无法验证环境"

cm_env="$(tr '\0' '\n' <"/proc/$cm_pid/environ")"
expected_environment=(
    'GTK_IM_MODULE=ibus'
    'QT_IM_MODULE=ibus'
    'XMODIFIERS=@im=ibus'
    "DBUS_SESSION_BUS_ADDRESS=$DBUS_ADDRESS"
)

for expected in "${expected_environment[@]}"; do
    printf '%s\n' "$cm_env" | grep -Fqx "$expected" || die "RustDesk --cm 未继承：$expected"
done

final_engine="$(ibus engine)"
[ "$final_engine" = "libpinyin" ] || die "当前 IBus 引擎不是 libpinyin，而是：$final_engine"

printf '\n修复完成。\n'
printf 'IBus 引擎：%s\n' "$final_engine"
printf 'RustDesk --cm PID：%s\n' "$cm_pid"
printf '请重新连接后，在 RustDesk 键盘模式中选择 Translate，再测试 Ctrl+Space。\n'
