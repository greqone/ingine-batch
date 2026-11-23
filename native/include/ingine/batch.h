// Batched generation: many requests, each in a slot of its own, decoded together (model::BatchEngine). The
// scheduler (engine::BatchScheduler) admits requests into free slots and steps every running one at once.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ingine::model {

class BatchBackend {
 public:
  // A request in a slot: `pending` is its last sampled token, not yet processed, at position `pos`.
  struct Seq {
    int slot;
    std::int32_t pending;
    std::int64_t pos;
    float temperature;  // <= 0: greedy
  };

  virtual ~BatchBackend() = default;
  virtual std::string model_id() const = 0;
  virtual int slots() const = 0;
  // Positions of a slot: a request's prompt, its output and the rows a step writes past its pending token
  // (lookahead()) must fit.
  virtual int ctx() const = 0;
  virtual int lookahead() const = 0;

  // Requests entering slots: each prompt is prefilled into its slot (whatever the slot held is dropped) and the
  // first output token sampled; a request continues as Seq{slot, that token, prompt.size(), temperature}.
  struct Admission {
    int slot;
    std::span<const std::int32_t> prompt;
    float temperature;
  };
  virtual std::vector<std::int32_t> admit(std::span<const Admission> requests) = 0;
  // One step over `seqs` (distinct slots): every pending token is processed; emitted[i] receives the tokens the
  // step produces for seqs[i] (one, or several with speculative decoding), and seqs[i] advances to the last of
  // them (pending) at its position.
  virtual void step(std::span<Seq> seqs, std::vector<std::vector<std::int32_t>>& emitted) = 0;
};

}  // namespace ingine::model
