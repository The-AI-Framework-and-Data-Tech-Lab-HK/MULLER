/*
 * Copyright (c) 2026 Bingyu Liu. All rights reserved.

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <fstream>
#include <string>
#include <mutex>
#include <iostream>

namespace hashmap {

class Logger {
public:
    enum LogLevel {
        DEBUG,
        INFO,
        WARNING,
        ERROR
    };

    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    // 初始化日志文件
    bool init(const std::string& logFilePath, LogLevel level = INFO) {
//        std::lock_guard<std::mutex> lock(m_mutex);

        // 如果已经初始化且路径相同，只更新日志级别
        if (m_initialized && m_logFilePath == logFilePath) {
            m_logLevel = level;
            return true;
        }

        // 关闭现有日志文件（如果已打开）
        if (m_initialized) {
            m_logFile.close();
        }
        m_logLevel = level;
        m_logFilePath = logFilePath;

        try {
            m_logFile.open(logFilePath, std::ios::out | std::ios::app);
            if (!m_logFile.is_open()) {
                std::cerr << "无法打开日志文件: " << logFilePath << std::endl;
                return false;
            }
            m_initialized = true;
            log(INFO, "日志系统初始化成功");
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "初始化日志系统时出错: " << e.what() << std::endl;
            return false;
        }
    }

    // 关闭日志文件
    void close() {
//        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_initialized) {
            m_logFile.close();
            m_initialized = false;
        }
    }

    // 记录日志的主要方法
    void log(LogLevel level, const std::string& message) {
        if (level < m_logLevel) return;

//        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_initialized) {
            std::cerr << getLevelString(level) << ": " << message << std::endl;
            return;
        }

        // 获取当前时间
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        char timeStr[26];
        ctime_r(&time, timeStr);
        timeStr[24] = '\0'; // 移除换行符

        // 写入日志
        m_logFile << "[" << timeStr << "] " << getLevelString(level) << ": " << message << std::endl;
        m_logFile.flush();

        // 同时输出到控制台（可选）
        #ifdef DEBUG_MODE
        std::cout << "[" << timeStr << "] " << getLevelString(level) << ": " << message << std::endl;
        #endif
    }

    // 便利方法
    void debug(const std::string& message) { log(DEBUG, message); }
    void info(const std::string& message) { log(INFO, message); }
    void warning(const std::string& message) { log(WARNING, message); }
    void error(const std::string& message) { log(ERROR, message); }

private:
    Logger() : m_initialized(false), m_logLevel(INFO) {}
    ~Logger() { close(); }

    // 禁止复制
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::string getLevelString(LogLevel level) {
        switch (level) {
            case DEBUG: return "DEBUG";
            case INFO: return "INFO";
            case WARNING: return "WARNING";
            case ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }

    bool m_initialized;
    std::ofstream m_logFile;
    std::mutex m_mutex;
    LogLevel m_logLevel;
    std::string m_logFilePath;
};

#define LOG_DEBUG(msg) Logger::getInstance().debug(msg)
#define LOG_INFO(msg) Logger::getInstance().info(msg)
#define LOG_WARNING(msg) Logger::getInstance().warning(msg)
#define LOG_ERROR(msg) Logger::getInstance().error(msg)

} // namespace hashmap