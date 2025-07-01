#pragma once

#ifdef ENABLE_LOGGING

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <memory>
#include <cstring>

namespace logger
{
  inline std::shared_ptr<spdlog::logger> &getSpdLogger()
  {
    static std::shared_ptr<spdlog::logger> logger = []() {
      // Create a file sink that truncates the file on each run
      auto file_sink
        = std::make_shared<spdlog::sinks::basic_file_sink_mt>("log.txt", true);
      file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");

      auto file_logger = std::make_shared<spdlog::logger>("file_logger", file_sink);
      file_logger->set_level(spdlog::level::info);
      spdlog::register_logger(file_logger);

      return file_logger;
    }();
    return logger;
  }
}

#define __FILENAME__                                                                     \
  (std::strrchr(__FILE__, '/') ? std::strrchr(__FILE__, '/') + 1 : __FILE__)
#define LOG_INFO(fmt, ...)                                                               \
  logger::getSpdLogger()->info("[{}:{}] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)                                                               \
  logger::getSpdLogger()->warn("[{}:{}] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...)                                                              \
  logger::getSpdLogger()->error("[{}:{}] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)

#endif // ENABLE_LOGGING
