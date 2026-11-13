# Critic learning-rate change awaiting explicit approval

The current run uses Actor/WM/Critic initial learning rates2e-4. At iteration349,
Critic loss reaches448.6768, mean value-64.6209 and mean TD target-60.2372, while
mean imagined reward is+0.0542. A frozen-snapshot diagnosis independently checks
TD(lambda) per trajectory and valid masks, with zero optimizer updates.
The failed stability evidence is preserved; finite values do not mean stable learning.

A separate implementation defect was found: the inherited PolicyTrainer.set_lr
assigns the Actor scheduled rate to BOTH optimizers. Therefore an explicit
critic_lr configuration is overwritten on the first scheduled update. The safe
interface fix is an HOI-local override retaining critic_lr/lr; Robot-only behavior
and the current equal-rate default remain unchanged. This fix does not itself
resolve the observed equal-rate instability.

Proposed experiment, NOT executed or approved yet:

- Keep Actor/WM initial lr2e-4; set ONLY Critic initial lr2e-5, retaining the ratio
  under the original cosine schedule and original Adam/TD(lambda)/BPTT updates.
- New scratch run, seed0,64 environments, batch256, H32,10 WM minibatches and50
  internal WM-only updates. No checkpoint state migration. Keep reference, assets,
  reward, network architecture, solver50, thresholds and all checks unchanged.
- Same long schedule target320000; bounded max_hours0.105, log interval50 and
  checkpoint interval1000. Expect approximately350 vector steps. Then perform
  strict same-config resume and physical play. Limit all work to the remaining
  original3600-second cumulative GPU budget; estimated additional time≤15 minutes.
- Compare value/target/loss/gradient trajectories, prediction errors, actual Actor
  episode and resources. Preserve failed candidate evidence. A smaller finite loss
  alone does not establish task learning or physical validity.
- Only if the comparison supports it, propose/promote critic_lr2e-5 as the documented
  full-train candidate default and synchronize the shell/README/resume contract.

Approval request covers this single parameter experiment and supported default
promotion only. Reference correction remains unexecuted; its separate feasibility
proposal is RESET_GEOMETRY_DISPOSITION.md. No reference/asset change is requested
as part of this learning-rate experiment.

Automatic approval review rejected the combined setter-fix/default-change action
because the Critic hyperparameter change was treated as a protected training
change requiring explicit user confirmation. No part of that rejected shell
command executed. This proposal is for review, not a workaround to run it.
