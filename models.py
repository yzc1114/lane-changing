import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import gym
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO, DDPG, A2C, TD3
from env import make_env, ObsType

from configuration import Configurable


class BaseModule(torch.nn.Module):
    """
        Base torch.nn.Module implementing basic features:
            - initialization factory
            - normalization parameters
    """
    def __init__(self, activation_type="RELU", reset_type="XAVIER", normalize=None):
        super().__init__()
        self.activation = activation_factory(activation_type)
        self.reset_type = reset_type
        self.normalize = normalize
        self.mean = None
        self.std = None

    def _init_weights(self, m):
        if hasattr(m, 'weight'):
            if self.reset_type == "XAVIER":
                torch.nn.init.xavier_uniform_(m.weight.data)
            elif self.reset_type == "ZEROS":
                torch.nn.init.constant_(m.weight.data, 0.)
            else:
                raise ValueError("Unknown reset type")
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.)

    def set_normalization_params(self, mean, std):
        if self.normalize:
            std[std == 0.] = 1.
        self.std = std
        self.mean = mean

    def reset(self):
        self.apply(self._init_weights)

    def forward(self, *input):
        if self.normalize:
            input = (input.float() - self.mean.float()) / self.std.float()
        return NotImplementedError


class MultiLayerPerceptron(BaseModule, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        sizes = [self.config["in"]] + self.config["layers"]
        self.activation = activation_factory(self.config["activation"])
        layers_list = [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.layers = nn.ModuleList(layers_list)
        if self.config.get("out", None):
            self.predict = nn.Linear(sizes[-1], self.config["out"])

    @classmethod
    def default_config(cls):
        return {"in": None,
                "layers": [64, 64],
                "activation": "RELU",
                "reshape": "True",
                "out": None}

    def forward(self, x):
        if self.config["reshape"]:
            x = x.reshape(x.shape[0], -1)  # We expect a batch of vectors
        for layer in self.layers:
            x = self.activation(layer(x))
        if self.config.get("out", None):
            x = self.predict(x)
        return x


class DuelingNetwork(BaseModule, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.config["base_module"]["in"] = self.config["in"]
        self.base_module = model_factory(self.config["base_module"])
        self.advantage = nn.Linear(self.config["base_module"]["layers"][-1], self.config["out"])
        self.value = nn.Linear(self.config["base_module"]["layers"][-1], 1)

    @classmethod
    def default_config(cls):
        return {"in": None,
                "base_module": {"type": "MultiLayerPerceptron", "out": None},
                "out": None}

    def forward(self, x):
        x = self.base_module(x)
        advantage = self.advantage(x)
        value = self.value(x).expand(-1,  self.config["out"])
        return value + advantage - advantage.mean(1).unsqueeze(1).expand(-1,  self.config["out"])


class ConvolutionalNetwork(nn.Module, Configurable): #3个卷积层+一个线性层
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.activation = activation_factory(self.config["activation"])
        self.conv1 = nn.Conv2d(self.config["in_channels"], 16, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=2)

        # MLP Head
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=2, stride=2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.config["in_width"])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.config["in_height"])))
        assert convh > 0 and convw > 0
        self.config["head_mlp"]["in"] = convw * convh * 64
        self.config["head_mlp"]["out"] = self.config["out"]
        self.head = model_factory(self.config["head_mlp"])

    @classmethod
    def default_config(cls):
        return {
            "in_channels": None,
            "in_height": None,
            "in_width": None,
            "activation": "RELU",
            "head_mlp": {
                "type": "MultiLayerPerceptron",
                "in": None,
                "layers": [],
                "activation": "RELU",
                "reshape": "True",
                "out": None
            },
            "out": None
        }

    def forward(self, x):
        """
            Forward convolutional network
        :param x: tensor of shape BCHW
        """
        x = self.activation((self.conv1(x)))
        x = self.activation((self.conv2(x)))
        x = self.activation((self.conv3(x)))
        return self.head(x)


class EgoAttention(BaseModule, Configurable): 
    #以ego作为主题，先将ego(1,fearures)和others(n-1,fearures)过一遍encoder(key_all,value_all,query_ego),再过注意力层(分成多个头，求qk^t内积 ->(1,entity)，乘value() (1,entity)*(entity*feature)->(1,fearures);),
    #再过decoder(1,fearures) * (fearures,fearures)->(1,fearures),再加上原来的ego除以2做残差，得到结果
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.features_per_head = int(self.config["feature_size"] / self.config["heads"])

        self.value_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.key_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.query_ego = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.attention_combine = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)

    @classmethod
    def default_config(cls):
        return {
            "feature_size": 64,
            "heads": 4,
            "dropout_factor": 0,
        }

    def forward(self, ego, others, mask=None):
        batch_size = others.shape[0]
        n_entities = others.shape[1] + 1
        input_all = torch.cat((ego.view(batch_size, 1, self.config["feature_size"]), others), dim=1)
        # Dimensions: Batch, entity, head, feature_per_head
        key_all = self.key_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)
        value_all = self.value_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)
        query_ego = self.query_ego(ego).view(batch_size, 1, self.config["heads"], self.features_per_head)

        # Dimensions: Batch, head, entity, feature_per_head
        key_all = key_all.permute(0, 2, 1, 3)
        value_all = value_all.permute(0, 2, 1, 3)
        query_ego = query_ego.permute(0, 2, 1, 3)
        if mask is not None:
            mask = mask.view((batch_size, 1, 1, n_entities)).repeat((1, self.config["heads"], 1, 1))
        value, attention_matrix = attention(query_ego, key_all, value_all, mask,
                                            nn.Dropout(self.config["dropout_factor"]))
        result = (self.attention_combine(value.reshape((batch_size, self.config["feature_size"]))) + ego.squeeze(1))/2 #残差
        return result, attention_matrix


class SelfAttention(BaseModule, Configurable):
    #与egoattention类似，只不过ego变成了all,最后得到(entity,fearures)
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.features_per_head = int(self.config["feature_size"] / self.config["heads"])

        self.value_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.key_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.query_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.attention_combine = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)

    @classmethod
    def default_config(cls):
        return {
            "feature_size": 64,
            "heads": 4,
            "dropout_factor": 0,
        }

    def forward(self, ego, others, mask=None):
        batch_size = others.shape[0]
        n_entities = others.shape[1] + 1
        input_all = torch.cat((ego.view(batch_size, 1, self.config["feature_size"]), others), dim=1)
        # Dimensions: Batch, entity, head, feature_per_head
        key_all = self.key_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)
        value_all = self.value_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)
        query_all = self.query_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)

        # Dimensions: Batch, head, entity, feature_per_head
        key_all = key_all.permute(0, 2, 1, 3)
        value_all = value_all.permute(0, 2, 1, 3)
        query_all = query_all.permute(0, 2, 1, 3)
        if mask is not None:
            mask = mask.view((batch_size, 1, 1, n_entities)).repeat((1, self.config["heads"], 1, 1))
        value, attention_matrix = attention(query_all, key_all, value_all, mask,
                                            nn.Dropout(self.config["dropout_factor"]))
        result = (self.attention_combine(value.reshape((batch_size, n_entities, self.config["feature_size"]))) + input_all)/2
        return result, attention_matrix


class EgoAttentionNetwork(BaseModule, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.config = config
        if not self.config["embedding_layer"]["in"]:
            self.config["embedding_layer"]["in"] = self.config["in"]
        if not self.config["others_embedding_layer"]["in"]:
            self.config["others_embedding_layer"]["in"] = self.config["in"]
        self.config["output_layer"]["in"] = self.config["attention_layer"]["feature_size"]
        self.config["output_layer"]["out"] = self.config["out"]

        self.ego_embedding = model_factory(self.config["embedding_layer"])
        self.others_embedding = model_factory(self.config["others_embedding_layer"])
        self.self_attention_layer = None #初始self_attention layer是none，如果不是none，embedding后的数据要先过一遍self_attention作为encoder(和ego里的encoder不一样)
        if self.config["self_attention_layer"]:
            self.self_attention_layer = SelfAttention(self.config["self_attention_layer"])
        self.attention_layer = EgoAttention(self.config["attention_layer"])
        self.output_layer = model_factory(self.config["output_layer"])

    @classmethod
    def default_config(cls):
        return {
            "in": None,
            "out": None,
            "presence_feature_idx": 0,
            "embedding_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128],
                "reshape": False
            },
            "others_embedding_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128],
                "reshape": False
            },
            "self_attention_layer": {
                "type": "SelfAttention",
                "feature_size": 128,
                "heads": 4
            },
            "attention_layer": {
                "type": "EgoAttention",
                "feature_size": 128,
                "heads": 4
            },
            "output_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 32],
                "reshape": False
            },
        }

    def forward(self, x):
        ego_embedded_att, _ = self.forward_attention(x)
        return self.output_layer(ego_embedded_att)

    def split_input(self, x, mask=None):
        # Dims: batch, entities, features
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        ego = x[:, 0:1, :]
        others = x[:, 1:, :]
        if mask is None:
            mask = x[:, :, self.config["presence_feature_idx"]:self.config["presence_feature_idx"] + 1] < 0.5
        return ego, others, mask

    def forward_attention(self, x):
        ego, others, mask = self.split_input(x)
        ego, others = self.ego_embedding(ego), self.others_embedding(others)
        if self.self_attention_layer:
            self_att, _ = self.self_attention_layer(ego, others, mask)
            ego, others, mask = self.split_input(self_att, mask=mask)
        return self.attention_layer(ego, others, mask)

    def get_attention_matrix(self, x):
        _, attention_matrix = self.forward_attention(x) #得到注意力矩阵
        return attention_matrix


class AttentionNetwork(BaseModule, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.config = config
        if not self.config["embedding_layer"]["in"]:
            self.config["embedding_layer"]["in"] = self.config["in"]
        self.config["output_layer"]["in"] = self.config["attention_layer"]["feature_size"]
        self.config["output_layer"]["out"] = self.config["out"]

        self.embedding = model_factory(self.config["embedding_layer"])
        self.attention_layer = SelfAttention(self.config["attention_layer"])
        self.output_layer = model_factory(self.config["output_layer"])

    @classmethod
    def default_config(cls):
        return {
            "in": None,
            "out": None,
            "presence_feature_idx": 0,
            "embedding_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128],
                "reshape": False
            },
            "attention_layer": {
                "type": "SelfAttention",
                "feature_size": 128,
                "heads": 4
            },
            "output_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128],
                "reshape": False
            },
        }

    def forward(self, x):
        ego, others, mask = self.split_input(x)
        ego_embedded_att, _ = self.attention_layer(self.embedding(ego), self.others_embedding(others), mask)
        return self.output_layer(ego_embedded_att)

    def split_input(self, x):
        # Dims: batch, entities, features
        ego = x[:, 0:1, :]
        others = x[:, 1:, :]
        mask = x[:, :, self.config["presence_feature_idx"]:self.config["presence_feature_idx"] + 1] < 0.5
        return ego, others, mask

    def get_attention_matrix(self, x):
        ego, others, mask = self.split_input(x)
        _, attention_matrix = self.attention_layer(self.embedding(ego), self.others_embedding(others), mask)
        return attention_matrix


def attention(query, key, value, mask=None, dropout=None):
    """
        Compute a Scaled Dot Product Attention.
    :param query: size: batch, head, 1 (ego-entity), features
    :param key:  size: batch, head, entities, features
    :param value: size: batch, head, entities, features
    :param mask: size: batch,  head, 1 (absence feature), 1 (ego-entity)
    :param dropout:
    :return: the attention softmax(QK^T/sqrt(dk))V
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output, p_attn


def activation_factory(activation_type):
    if activation_type == "RELU":
        return F.relu
    elif activation_type == "TANH":
        return torch.tanh
    else:
        raise ValueError("Unknown activation_type: {}".format(activation_type))


def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def size_model_config(env, model_config):
    """
        Update the configuration of a model depending on the environment observation/action spaces

        Typically, the input/output sizes.

    :param env: an environment
    :param model_config: a model configuration
    """
    if model_config["type"] == "ConvolutionalNetwork":  # Assume CHW observation space
        model_config["in_channels"] = int(env.observation_space.shape[0])
        model_config["in_height"] = int(env.observation_space.shape[1])
        model_config["in_width"] = int(env.observation_space.shape[2])
    else:
        model_config["in"] = int(np.prod(env.observation_space.shape))
    model_config["out"] = env.action_space.n


def model_factory(config: dict) -> nn.Module:
    '''
    Returns the model based on the given config's type
    '''
    if config["type"] == "MultiLayerPerceptron":
        return MultiLayerPerceptron(config)
    elif config["type"] == "DuelingNetwork":
        return DuelingNetwork(config)
    elif config["type"] == "ConvolutionalNetwork":
        return ConvolutionalNetwork(config)
    elif config["type"] == "EgoAttentionNetwork":
        return EgoAttentionNetwork(config)
    # elif config["type"] == "head_mlp":
    #     return MultiLayerPerceptron(config)
    else:
        raise ValueError("Unknown model type")

EgoAttentionNetwork_config = {
            "in": None,
            "out": None,
            "presence_feature_idx": 0,
            "embedding_layer": {
                "in": None,
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128],
                "out": None,
                "reshape": False
            },
            "others_embedding_layer": {
                "in": None,
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128],
                "out": None,
                "reshape": False
            },
            "self_attention_layer": {
                "type": "SelfAttention",
                "feature_size": 128,
                "heads": 4
            },
            "attention_layer": {
                "type": "EgoAttention",
                "feature_size": 128,
                "heads": 4
            },
            "output_layer": {
                "in": None,
                "type": "MultiLayerPerceptron",
                "layers": [128,128],
                "out": None,
                "reshape": False
            },
        }

DuelNetwork_config = {"in": 32,
                "base_module": {"type": "MultiLayerPerceptron", "out": None},
                "out": 5}


class EgoAttentionNetwork_feature_extractor(
                            # EgoAttentionNetwork,
                            BaseFeaturesExtractor):
    def __init__(self,observation_space: gym.spaces.Box, ego_config = EgoAttentionNetwork_config,duel_config = DuelNetwork_config,features_dim = 5):
        # super(EgoAttentionNetwork_feature_extractor,self).__init__(self,observation_space,features_dim = 32)
        # EgoAttentionNetwork.__init__(self,config=config)
        super(EgoAttentionNetwork_feature_extractor,self).__init__(observation_space,features_dim=features_dim)
        self.ego_config = ego_config
        self.duel_config = duel_config

        self.ego_config["in"] = observation_space.shape[1] #spaces.Box(shape=(self.vehicles_count, len(self.features))  
        # self.config["in"] = int(np.prod(observation_space.shape))  
        self.ego_config["out"] = features_dim

        self.duel_config["in"] = features_dim
        self.egoAttentionNetwork = EgoAttentionNetwork(config=self.ego_config)
        self.duelnetwork = DuelingNetwork(config=self.duel_config)

        # self.activation = nn.ReLU()





    def forward(self,observations: torch.Tensor)->torch.Tensor:
        ego_embedded_att, _ = self.egoAttentionNetwork.forward_attention(observations)
        ego_out = self.egoAttentionNetwork.output_layer(ego_embedded_att)
        x = self.duelnetwork(ego_out)
        return x


class GridOccupancyCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(GridOccupancyCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

def do_test():
    env = make_env(1, ObsType.OccupancyGrid)
    policy_kwargs = dict(
        features_extractor_class=GridOccupancyCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[256, dict(pi=[256, 64], vf=[256, 256])]
    )
    batch_size = 64
    model = PPO("MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=5e-4,
                n_steps=batch_size,
                batch_size=batch_size,
                n_epochs=10,
                gamma=0.8,
                verbose=2,
                tensorboard_log="./log/")
    model.learn(5000)


if __name__ == '__main__':
    do_test()