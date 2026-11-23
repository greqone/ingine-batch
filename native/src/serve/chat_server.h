// OpenAI-compatible HTTP server (mirror of engine/ingine/server.py App/Handler/ChatServer): POST
// /v1/chat/completions (streaming SSE and non-streaming), GET /v1/models[/<id>], GET /health.
//
// cpp-httplib gives each connection a pool thread that validates the request, renders the prompt and submits a
// Job to the job queue (engine::Scheduler: one request at a time; engine::BatchScheduler: many; full -> 503). The
// request thread drains the job's tokens, runs the StreamParser and writes the response; it polls the socket while
// waiting and cancels the job when the client disconnects (or a stop string / stop token ends the turn).
#pragma once

#include "engine/scheduler.h"
#include "ingine/generator.h"
#include "serve/openai.h"
#include "text/chat.h"

#include <httplib.h>

#include <cstdint>
#include <memory>
#include <string>
#include <thread>

namespace ingine::serve {

struct ServerOptions {
  int max_queue = 4;      // requests allowed to wait behind the running one(s)
  int max_model_len = 0;  // context limit; 0: generator.max_model_len()
  std::string model_id;   // model name reported by the API; empty: generator.model_id()
  model::Sampling sampling;  // temperature / top_p / top_k of requests that omit them
  ImageOptions images;       // image input; in effect only when the generator supports images
  int http_threads = 8;      // connection threads: a streaming request holds one until it ends
};

class ChatServer {
 public:
  ChatServer(model::Generator& generator, const text::ChatFormat& fmt, ServerOptions options = {});
  // Batched serving: requests go to `queue` (an engine::BatchScheduler); text only.
  ChatServer(std::unique_ptr<engine::JobQueue> queue, const std::string& model_id, int max_model_len,
             const text::ChatFormat& fmt, ServerOptions options = {});
  ~ChatServer();
  ChatServer(const ChatServer&) = delete;
  ChatServer& operator=(const ChatServer&) = delete;

  // Binds the listening socket (port 0: any free port, see port()).
  bool bind(const std::string& host, int port);
  int port() const { return port_; }
  // Blocking accept loop on the bound socket; returns after stop().
  bool listen();
  // Serves on a background thread (tests).
  void start();
  // Fails waiting jobs, cancels the running one, stops accepting and joins the background thread.
  void stop();

  const std::string& model_id() const { return model_id_; }
  int max_model_len() const { return max_model_len_; }

 private:
  void setup(int http_threads);
  void register_routes();
  void handle_get(const httplib::Request& req, httplib::Response& res);
  void handle_post(const httplib::Request& req, httplib::Response& res);
  void complete(const httplib::Request& req, httplib::Response& res, ChatRequest request);
  Json model_card() const;

  const text::ChatFormat& fmt_;
  const std::string model_id_;
  const int max_model_len_;
  const model::Sampling sampling_defaults_;
  const ImageOptions image_options_;
  const int64_t created_;
  std::unique_ptr<engine::JobQueue> queue_;
  httplib::Server server_;
  int port_ = 0;
  std::thread thread_;
};

}  // namespace ingine::serve
