"""PPO but without the adaptive KL term that RLlib added."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import ray
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule, ACTION_LOGP
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.tf_ops import make_tf_callable
from ray.rllib.utils import try_import_tf

from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.ppo.ppo import choose_policy_optimizer, DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo import warn_about_bad_reward_scales

tf = try_import_tf()

logger = logging.getLogger(__name__)

# Frozen logits of the policy that computed the action
BEHAVIOUR_LOGITS = "behaviour_logits"


class PPOLoss(object):
    """PPO Loss object."""

    def __init__(self,
                 action_space,
                 dist_class,
                 model,
                 value_targets,
                 advantages,
                 actions,
                 prev_logits,
                 prev_actions_logp,
                 vf_preds,
                 curr_action_dist,
                 value_fn,
                 cur_kl_coeff,
                 valid_mask,
                 entropy_coeff=0,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 use_gae=True,
                 model_config=None):
        """Construct the loss for Proximal Policy Objective.

        Parameters
        ----------
        action_space : TODO
            Environment observation space specification.
        dist_class : TODO
            action distribution class for logits.
        value_targets : tf.placeholder
            Placeholder for target values; used for GAE.
        actions : tf.placeholder
            Placeholder for actions taken from previous model evaluation.
        advantages : tf.placeholder
            Placeholder for calculated advantages from previous model
            evaluation.
        prev_logits : tf.placeholder
            Placeholder for logits output from previous model evaluation.
        prev_actions_logp : tf.placeholder
            Placeholder for prob output from previous model evaluation.
        vf_preds : tf.placeholder
            Placeholder for value function output from previous model
            evaluation.
        curr_action_dist : ActionDistribution
            ActionDistribution of the current model.
        value_fn : tf.Tensor
            Current value function output Tensor.
        cur_kl_coeff : tf.Variable
            Variable holding the current PPO KL coefficient.
        valid_mask : tf.Tensor
            A bool mask of valid input elements (#2992).
        entropy_coeff : float
            Coefficient of the entropy regularizer.
        clip_param : float
            Clip parameter
        vf_clip_param : float
            Clip parameter for the value function
        vf_loss_coeff : float
            Coefficient of the value function loss
        use_gae : bool
            If true, use the Generalized Advantage Estimator.
        model_config : dict, optional
            model config for use in specifying action distributions.
        """

        def reduce_mean_valid(t):
            return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

        prev_dist = dist_class(prev_logits, model)
        # Make loss functions.
        logp_ratio = tf.exp(curr_action_dist.logp(actions) - prev_actions_logp)
        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = reduce_mean_valid(action_kl)

        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = tf.minimum(
            advantages * logp_ratio,
            advantages * tf.clip_by_value(logp_ratio, 1 - clip_param,
                                          1 + clip_param))
        self.mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        if use_gae:
            vf_loss1 = tf.square(value_fn - value_targets)
            vf_clipped = vf_preds + tf.clip_by_value(
                value_fn - vf_preds, -vf_clip_param, vf_clip_param)
            vf_loss2 = tf.square(vf_clipped - value_targets)
            vf_loss = tf.maximum(vf_loss1, vf_loss2)
            self.mean_vf_loss = reduce_mean_valid(vf_loss)
            loss = reduce_mean_valid(
                -surrogate_loss +
                vf_loss_coeff * vf_loss - entropy_coeff * curr_entropy)
        else:
            self.mean_vf_loss = tf.constant(0.0)
            loss = reduce_mean_valid(-surrogate_loss - entropy_coeff * curr_entropy)
        self.loss = loss


def ppo_surrogate_loss(policy, model, dist_class, train_batch):
    """Construct and return the PPO loss."""
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(
            train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool)

    policy.loss_obj = PPOLoss(
        policy.action_space,
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[BEHAVIOUR_LOGITS],
        train_batch[ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        model.value_function(),
        policy.kl_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
        model_config=policy.config["model"])

    return policy.loss_obj.loss


def kl_and_loss_stats(policy, train_batch):
    """Return statistics for the tensorboard."""
    return {
        "cur_kl_coeff": tf.cast(policy.kl_coeff, tf.float64),
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        "total_loss": policy.loss_obj.loss,
        "policy_loss": policy.loss_obj.mean_policy_loss,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.model.value_function()),
        "vf_preds": train_batch[Postprocessing.VALUE_TARGETS],
        "kl": policy.loss_obj.mean_kl,
        "entropy": policy.loss_obj.mean_entropy,
        "entropy_coeff": tf.cast(policy.entropy_coeff, tf.float64),
        "advantages": train_batch[Postprocessing.ADVANTAGES],
        "rewards": train_batch["rewards"]
    }


def vf_preds_and_logits_fetches(policy):
    """Add value function and logits outputs to experience train_batches."""
    return {
        SampleBatch.VF_PREDS: policy.model.value_function(),
        BEHAVIOUR_LOGITS: policy.model.last_output(),
    }


def postprocess_ppo_gae(policy,
                        sample_batch,
                        other_agent_batches=None,
                        episode=None):
    """Add the policy logits, VF preds, and advantages to the trajectory."""
    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                               sample_batch[SampleBatch.ACTIONS][-1],
                               sample_batch[SampleBatch.REWARDS][-1],
                               *next_state)

    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return batch


def clip_gradients(policy, optimizer, loss):
    """If grad_clip is not None, clip the gradients."""
    variables = policy.model.trainable_variables()
    if policy.config["grad_clip"] is not None:
        grads_and_vars = optimizer.compute_gradients(loss, variables)
        grads = [g for (g, v) in grads_and_vars]
        policy.grads, _ = tf.clip_by_global_norm(grads,
                                                 policy.config["grad_clip"])
        clipped_grads = list(zip(policy.grads, variables))
        return clipped_grads
    else:
        return optimizer.compute_gradients(loss, variables)


class ValueNetworkMixin(object):
    """Construct the value function."""

    def __init__(self, obs_space, action_space, config):
        if config["use_gae"]:

            @make_tf_callable(self.get_session())
            def value(ob, prev_action, prev_reward, *state):
                model_out, _ = self.model({
                    SampleBatch.CUR_OBS: tf.convert_to_tensor([ob]),
                    SampleBatch.PREV_ACTIONS: tf.convert_to_tensor(
                        [prev_action]),
                    SampleBatch.PREV_REWARDS: tf.convert_to_tensor(
                        [prev_reward]),
                    "is_training": tf.convert_to_tensor(False),
                }, [tf.convert_to_tensor([s]) for s in state],
                    tf.convert_to_tensor([1]))
                return self.model.value_function()[0]

        else:

            @make_tf_callable(self.get_session())
            def value(ob, prev_action, prev_reward, *state):
                return tf.constant(0.0)

        self._value = value


def setup_config(policy, obs_space, action_space, config):
    """Add additional custom options from the config."""
    # auto set the model option for layer sharing
    config["model"]["vf_share_layers"] = config["vf_share_layers"]


def setup_mixins(policy, obs_space, action_space, config):
    """Construct additional classes that add on to PPO."""
    KLCoeffMixin.__init__(policy, config)
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


class KLCoeffMixin(object):
    """Update the KL Coefficient. This is intentionally disabled to match the PPO paper better."""

    def __init__(self, config):
        # KL Coefficient
        self.kl_coeff_val = config["kl_coeff"]
        self.kl_target = config["kl_target"]
        self.kl_coeff = tf.get_variable(
            initializer=tf.constant_initializer(self.kl_coeff_val),
            name="kl_coeff",
            shape=(),
            trainable=False,
            dtype=tf.float32)

    def update_kl(self, blah):
        """Disabled to match the PPO paper better."""
        pass


CustomPPOTFPolicy = build_tf_policy(
    name="CustomPPOTFPolicy",
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    extra_action_fetches_fn=vf_preds_and_logits_fetches,
    postprocess_fn=postprocess_ppo_gae,
    gradients_fn=clip_gradients,
    before_init=setup_config,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule,
        ValueNetworkMixin, KLCoeffMixin
    ])


def validate_config(config):
    """Check that the config is set up properly."""
    if config["entropy_coeff"] < 0:
        raise DeprecationWarning("entropy_coeff must be >= 0")
    if isinstance(config["entropy_coeff"], int):
        config["entropy_coeff"] = float(config["entropy_coeff"])
    if config["batch_mode"] == "truncate_episodes" and not config["use_gae"]:
        raise ValueError(
            "Episode truncation is not supported without a value "
            "function. Consider setting batch_mode=complete_episodes.")
    if config["multiagent"]["policies"] and not config["simple_optimizer"]:
        logger.info(
            "In multi-agent mode, policies will be optimized sequentially "
            "by the multi-GPU optimizer. Consider setting "
            "simple_optimizer=True if this doesn't work for you.")
    if config["simple_optimizer"]:
        logger.warning(
            "Using the simple minibatch optimizer. This will significantly "
            "reduce performance, consider simple_optimizer=False.")
    elif tf and tf.executing_eagerly():
        config["simple_optimizer"] = True  # multi-gpu not supported


CustomPPOTrainer = build_trainer(
    name="CustomPPOTrainer",
    default_config=DEFAULT_CONFIG,
    default_policy=CustomPPOTFPolicy,
    make_policy_optimizer=choose_policy_optimizer,
    validate_config=validate_config,
    after_train_result=warn_about_bad_reward_scales)
