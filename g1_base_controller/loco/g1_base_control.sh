#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMERGENCY_STOP_SCRIPT="${SCRIPT_DIR}/emergency_stop_monitor.py"
G1_CONTROLLER_SCRIPT="${SCRIPT_DIR}/g1_base_controller.py"

# PID文件
EMERGENCY_PID="/tmp/emergency_stop.pid"
G1_CONTROLLER_PID="/tmp/g1_controller.pid"

start() {
    echo "启动G1控制系统..."
    
    # 启动急停监控器
    echo "启动急停监控器..."
    python3 "$EMERGENCY_STOP_SCRIPT" &
    echo $! > "$EMERGENCY_PID"
    echo "急停监控器启动 (PID: $(cat $EMERGENCY_PID))"
    
    # 等待2秒再启动控制器
    sleep 2
    
    # 启动G1控制器
    echo "启动G1控制器..."
    python3 "$G1_CONTROLLER_SCRIPT" &
    echo $! > "$G1_CONTROLLER_PID"
    echo "G1控制器启动 (PID: $(cat $G1_CONTROLLER_PID))"
    
    echo "G1控制系统启动完成！"
}

stop() {
    echo "停止G1控制系统..."
    
    # 停止G1控制器
    if [ -f "$G1_CONTROLLER_PID" ]; then
        PID=$(cat "$G1_CONTROLLER_PID")
        if kill -0 "$PID" 2>/dev/null; then
            echo "停止G1控制器 (PID: $PID)"
            kill "$PID"
        fi
        rm -f "$G1_CONTROLLER_PID"
    fi
    
    # 停止急停监控器
    if [ -f "$EMERGENCY_PID" ]; then
        PID=$(cat "$EMERGENCY_PID")
        if kill -0 "$PID" 2>/dev/null; then
            echo "停止急停监控器 (PID: $PID)"
            kill "$PID"
        fi
        rm -f "$EMERGENCY_PID"
    fi
    
    echo "G1控制系统已停止！"
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        sleep 2
        start
        ;;
    *)
        echo "用法: $0 {start|stop|restart}"
        exit 1
        ;;
esac