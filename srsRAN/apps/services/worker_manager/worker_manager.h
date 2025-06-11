/*
 *
 * Copyright 2021-2025 Software Radio Systems Limited
 *
 * This file is part of srsRAN.
 *
 * srsRAN is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * srsRAN is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * A copy of the GNU Affero General Public License can be found in
 * the LICENSE file in the top-level directory of this distribution
 * and at http://www.gnu.org/licenses/.
 *
 */

#pragma once

#include "worker_manager_config.h"
#include "worker_manager_worker_getter.h"
#include "srsran/cu_up/cu_up_executor_mapper.h"
#include "srsran/du/du_high/du_high_executor_mapper.h"
#include "srsran/support/executors/task_execution_manager.h"
#include "srsran/support/executors/task_executor.h"

namespace srsran {

/// Manages the workers of the app.
struct worker_manager : public worker_manager_executor_getter {
  worker_manager(const worker_manager_config& config);

  void stop();

  /// du ctrl exec points to general ctrl_worker
  /// du ue exec points to the general ue_worker
  /// cu-cp ctrl exec points to general ctrl_worker (just like the du ctrl exec)
  /// cu-up ue exec points to the general ue_worker (just like the du ue exec)
  ///
  /// The handler side is responsible for executor dispatching:
  /// - ngap::handle_message calls cu-cp ctrl exec
  /// - f1ap_cu::handle_message calls cu-cp ctrl exec
  /// - e1ap_cu_cp::handle_message calls cu-cp ctrl exec
  /// - e1ap_cu_up::handle_message calls cu-up ue exec

  task_executor*              cu_cp_exec = nullptr;
  std::vector<task_executor*> lower_phy_tx_exec;
  std::vector<task_executor*> lower_phy_rx_exec;
  std::vector<task_executor*> lower_phy_dl_exec;
  std::vector<task_executor*> lower_phy_ul_exec;
  std::vector<task_executor*> lower_prach_exec;
  std::vector<task_executor*> upper_pusch_exec;
  std::vector<task_executor*> upper_pusch_decoder_exec;
  std::vector<task_executor*> upper_pucch_exec;
  std::vector<task_executor*> upper_prach_exec;
  std::vector<task_executor*> upper_srs_exec;
  std::vector<task_executor*> upper_pdsch_exec;
  task_executor*              radio_exec     = nullptr;
  task_executor*              ru_timing_exec = nullptr;
  std::vector<task_executor*> ru_txrx_exec;
  std::vector<task_executor*> fapi_exec;
  std::vector<task_executor*> ru_dl_exec;
  std::vector<task_executor*> ru_rx_exec;
  task_executor*              cu_e2_exec              = nullptr;
  task_executor*              non_rt_low_prio_exec    = nullptr;
  task_executor*              non_rt_medium_prio_exec = nullptr;
  task_executor*              non_rt_hi_prio_exec     = nullptr;
  task_executor*              split6_exec             = nullptr;
  task_executor*              metrics_exec            = nullptr;

  std::unique_ptr<srs_cu_up::cu_up_executor_mapper> cu_up_exec_mapper;

  srs_du::du_high_executor_mapper& get_du_high_executor_mapper() { return *du_high_exec_mapper; }

  // Gets the DU-low downlink executors.
  task_executor& get_du_low_dl_executor(unsigned sector_id) const;

  /// Get executor based on the name.
  task_executor* find_executor(const std::string& name) const
  {
    auto it = exec_mng.executors().find(name);
    return it != exec_mng.executors().end() ? it->second : nullptr;
  }

  task_executor& get_executor(const std::string& name) const override { return *exec_mng.executors().at(name); }

  worker_manager_executor_getter* get_executor_getter() { return this; }

private:
  struct du_crit_path_executor_desc {
    // Description of L2 executors for the DU-high.
    srs_du::du_high_executor_config::cell_executor_config l2_execs;
    // Description of L1 executors for DU-low.
    // TODO
  };
  std::unique_ptr<srs_du::du_high_executor_mapper> du_high_exec_mapper;
  std::vector<task_executor*>                      du_low_dl_executors;

  /// Manager of execution contexts and respective executors instantiated by the application.
  task_execution_manager exec_mng;

  os_sched_affinity_manager low_prio_affinity_mng;
  os_sched_affinity_bitmask ru_timing_mask;

  std::vector<os_sched_affinity_bitmask> ru_txrx_affinity_masks;

  /// CPU affinity bitmask manager per cell.
  std::vector<os_sched_affinity_manager> affinity_mng;

  /// Helper method to create workers with non zero priority.
  void create_prio_worker(const std::string&                                    name,
                          unsigned                                              queue_size,
                          const std::vector<execution_config_helper::executor>& execs,
                          const os_sched_affinity_bitmask&                      mask,
                          os_thread_realtime_priority prio = os_thread_realtime_priority::no_realtime());

  /// Helper method to create worker pool.
  void create_worker_pool(const std::string&                                    name,
                          unsigned                                              nof_workers,
                          unsigned                                              queue_size,
                          const std::vector<execution_config_helper::executor>& execs,
                          os_thread_realtime_priority           prio      = os_thread_realtime_priority::no_realtime(),
                          span<const os_sched_affinity_bitmask> cpu_masks = {},
                          concurrent_queue_policy               queue_policy = concurrent_queue_policy::locking_mpmc);

  execution_config_helper::worker_pool create_low_prio_workers(unsigned                  nof_low_prio_threads,
                                                               unsigned int              low_prio_task_queue_size,
                                                               os_sched_affinity_bitmask low_prio_mask);
  void                                 create_low_prio_executors(const worker_manager_config& config);
  void                                 associate_low_prio_executors(const worker_manager_config& config);

  std::vector<execution_config_helper::single_worker> create_fapi_workers(unsigned nof_cells);

  /// Helper method that creates the CU-UP executors.
  void create_cu_up_executors(const worker_manager_config::cu_up_config& config, timer_manager& timers);

  /// Helper method that creates the Distributed Unit executors.
  void create_du_executors(const worker_manager_config::du_high_config&        du_hi,
                           std::optional<worker_manager_config::du_low_config> du_low,
                           std::optional<worker_manager_config::fapi_config>   fapi_cfg,
                           timer_manager&                                      timers);

  /// Helper method that creates the low Distributed Unit executors.
  du_crit_path_executor_desc
  create_du_crit_path_prio_executors(unsigned                                                   nof_cells,
                                     bool                                                       rt_mode,
                                     const std::optional<worker_manager_config::du_low_config>& du_low);

  /// Helper method that creates the Radio Unit dummy executors.
  void create_ru_dummy_executors();

  /// Helper method that creates the lower PHY executors.
  void create_lower_phy_executors(const worker_manager_config::ru_sdr_config& config);

  /// Helper method that creates the Open Fronthaul executors.
  void create_ofh_executors(const worker_manager_config::ru_ofh_config& config);

  /// Helper method that creates the split 6 executors.
  void create_split6_executors();
};

} // namespace srsran
