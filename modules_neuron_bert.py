from torch import nn
import torch.nn.functional as F
import torch
from spikingjelly.activation_based import neuron_bert
from timm.utils import AverageMeter


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


# class MCN(nn.Module):
#     def __init__(self, up=2., t=32, batch_size=64):
#         super().__init__()
#         self.max_act = 0
#         self.t = t
#         self.batch_size = batch_size
#         self.relu_bool = False
#         self.i_layer = 0
#         self.relu = nn.ReLU()
#         self.name = None
#         self.full_scale = False
#         self.batch_fusion_bool = False

#     def forward(self, x):

#         if (self.relu_bool):
#             x = self.relu(x)
#         ######
#         # Code for get distribution of Activation on Transformer
#         ######

#         # find the scale of each neuron_bert on inference
#         if (isinstance(self.max_act, int)):
#             if (x.shape[0] != self.batch_size):
#                 self.batch_fusion_bool = True
#                 tp_x = x.clone()
#                 tp_shape = list(x.size())
#                 tp_shape[0] = int(tp_shape[0]/self.batch_size)

#                 tp_shape = list([self.batch_size])+tp_shape
#                 tp_x = tp_x.view(tp_shape)
#                 if (len(tp_shape) < 5):
#                     self.max_act = tp_x.abs().max(0).values
#                     self.full_scale = True
#                 else:
#                     self.max_act = tp_x.abs().max(0).values.max(-1).values
#             else:
#                 if (len(x.shape) < 4):
#                     self.max_act = x.abs().max(0).values
#                     self.full_scale = True
#                 else:
#                     self.max_act = x.abs().max(0).values.max(-1).values
#         else:
#             if (self.batch_fusion_bool):
#                 tp_x = x.clone()
#                 tp_shape = list(x.shape)
#                 batch_size_tp = x.shape[0]//self.max_act.shape[0]
#                 tp_shape[0] = int(tp_shape[0]/batch_size_tp)
#                 tp_shape = list([batch_size_tp])+tp_shape
#                 tp_x = tp_x.view(tp_shape)
#                 if (self.full_scale):
#                     self.max_act = torch.maximum(
#                         tp_x.abs().max(0).values, self.max_act)
#                 else:
#                     self.max_act = torch.maximum(tp_x.abs().max(
#                         0).values.max(-1).values, self.max_act)

#             else:

#                 if (self.full_scale):
#                     self.max_act = torch.maximum(
#                         x.abs().max(0).values, self.max_act)

#                 else:
#                     self.max_act = torch.maximum(x.abs().max(
#                         0).values.max(-1).values, self.max_act)

#         return x






class MCN(nn.Module):
    def __init__(self, up=2., t=32, batch_size=64):
        super().__init__()
        self.max_act = 0
        self.t = t
        self.batch_size = batch_size
        self.relu_bool = False
        self.i_layer = 0
        self.relu = nn.ReLU()
        self.name = None
        self.full_scale = False
        self.batch_fusion_bool = False
        self.last_dim =0
        self.sparisty_meter = AverageMeter()

    def forward(self, x):

        if (self.relu_bool):
            x = self.relu(x)
        

        ######
        # Code for get distribution of Activation on Transformer
        ######

        # find the scale of each neuron_bert on inference
        if (isinstance(self.max_act, int)):
                    if(self.name=="stdp_av" or self.name=="stdp_qk" or self.name=="k_if" or self.name=="q_if" or self.name=="v_if" ):
                        self.max_act = x.abs().max(0).values.max(-1).values.unsqueeze(-1)
                        self.last_dim = x.shape[-1]

                    else:
                        self.max_act = x.abs().max(0).values
        else:
                    if(self.name=="stdp_av" or self.name=="stdp_qk"or self.name=="k_if" or self.name=="q_if" or self.name=="v_if" ):
                        self.max_act = torch.maximum(x.abs().max(0).values.max(-1).values.unsqueeze(-1), self.max_act)
                        # self.last_dim = x.shape[-1]

                    else:
                        self.max_act = torch.maximum(x.abs().max(0).values, self.max_act)
        self.sparisty_meter.update(torch.count_nonzero(x)/torch.numel(x))

        return x


class MCN_2(nn.Module):
    def __init__(self, base=2., timestep=16, batch_size=64,max_scale= None,relu_bool=False, softmax_bool= False, name=None,n_layer=None):
        super().__init__()
        self.max_act = 0
        self.batch_size = batch_size
        self.relu_bool = relu_bool
        self.i_layer = n_layer
        self.relu = nn.ReLU()
        self.name = name
        self.full_scale = False
        self.batch_fusion_bool = False
        self.base = base
        self.timestep = timestep
        self.scale = max_scale
        self.softmax_bool = softmax_bool

    def forward(self, x):
        assert x[0].shape==self.scale.shape,(x.shape,self.scale.shape )
        if (self.relu_bool):
            x = self.relu(x)



        if(self.softmax_bool):
            atten_scale = x.max(-1).values.unsqueeze(-1)
            attn = x/atten_scale
            tp_round = (-torch.round(( 1-attn)/(torch.log(torch.tensor(self.base))/atten_scale ))) #*(1/torch.log(torch.tensor(base)))
            
            attn = torch.where(tp_round>=-self.timestep,self.base**tp_round,0)
            result_x = attn/attn.sum(-1).unsqueeze(-1)
        else:
            x_ab = x.abs()
            x_sign = x.sign()
            # scale_v =  v_ab.max(-1).values.unsqueeze(-1)
            x_ab = x_ab/self.scale
            tp_log_value = torch.log(x_ab)/torch.log(torch.tensor(self.base))

            tp_round = torch.where(tp_log_value<=-1,torch.round(tp_log_value),-1)
            result_x = torch.where(tp_round>=-self.timestep,self.base**tp_round,0)*x_sign*self.scale



        return result_x


class MCN_3(nn.Module):
    def __init__(self, base=2., timestep=16, batch_size=64,max_scale= None,relu_bool=False, softmax_bool= False, name=None,n_layer=None):
        super().__init__()
        self.max_act = max_scale
        self.batch_size = batch_size
        self.relu_bool = relu_bool
        self.i_layer = n_layer
        self.relu = nn.ReLU()
        self.name = name
        self.full_scale = False
        self.batch_fusion_bool = False
        self.base = base
        self.timestep = timestep
        self.scale = max_scale
        self.softmax_bool = softmax_bool
        self.last_dim =0
        self.sparisty_meter = AverageMeter()
        self.timstep_cycle =0
        if(n_layer==0):
            self.convert_bool = False 
        else:
            self.convert_bool = False
        

        if(softmax_bool):
            self.neuron_bert=WTA_layer_neuron_bert(scale=1.0, timestep=timestep, wait=timestep, start_time=0, i_layer=n_layer, tau=base, convert=False, modulename=name, trace_bool=True)
                    
        else:
            if(self.relu_bool):
                self.neuron_bert = Scaledneuron_bert_onespike_time_relu(scale=self.scale, timestep=timestep, wait=1, start_time=0, i_layer=n_layer,
                                                        tau=base, convert=self.convert_bool, trace_bool=True, stdp_bool=False)   
            
            else:
                self.neuron_bert = Scaledneuron_bert_onespike_time_bipolar(scale=self.scale, timestep=timestep, wait=1, start_time=0, i_layer=n_layer,
                                                        tau=base, convert=self.convert_bool, trace_bool=True, stdp_bool=False)   
            
    def forward(self, x):
        # assert x[0].shape==self.scale.shape,(x.shape,self.scale.shape )
        # if (self.relu_bool):
        #     x = self.relu(x)

        
        if(self.softmax_bool):
            for t in range(self.timestep):
                a= self.neuron_bert(x[t])
            for t in range(self.timestep):    
                if(t==0):
                        result_x =self.neuron_bert(None)
                        # assert False, (result_x,self.neuron_bert.t)
                else:
                        result_x *=self.base
                        result_x +=self.neuron_bert(None)
            result_x= result_x/(self.base**(self.timestep))/self.neuron_bert.spike_sum.unsqueeze(-1)
            self.neuron_bert.reset()
            
        elif(self.name=="stdp_qk"):
            a= self.neuron_bert(x)
            result_x = []
            for t in range(self.timestep):    
                result =self.neuron_bert(None)   
                result_x.append(result)
            self.neuron_bert.reset()
        else:
            a= self.neuron_bert(x)
            for t in range(self.timestep):    
                if(t==0):
                    result_x =self.neuron_bert(None)
                    # assert False, (result_x,self.neuron_bert.t)
                else:
                    result_x *=self.base
                    result_tp =self.neuron_bert(None)
                    result_x +=result_tp
            result_x= result_x/(self.base**(self.timestep-1))
            self.neuron_bert.reset()

        if(not self.name=="stdp_qk"):
            self.sparisty_meter.update(torch.count_nonzero(result_x)/torch.numel(result_x))

        self.timstep_cycle = self.neuron_bert.mem_count_meter.avg
        return result_x




class Scaledneuron_bert_onespike_time_relu(nn.Module):
    def __init__(self, scale=1., timestep=24, wait=12, start_time=0, i_layer=0, tau=2.0, convert=False, modulename=None, trace_bool=False, stdp_bool=False, scale_full=False, final_bool=False):
        super(Scaledneuron_bert_onespike_time_relu, self).__init__()
        self.scale_full = scale_full

        self.final_bool = final_bool
        self.scale = scale
        self.timestep = timestep
        self.wait = wait
        self.starttime = start_time
        self.t = 0
        self.convert = convert
        self.block = 0
        self.module_name = modulename
        self.i_layer = i_layer
        self.stdp_bool = stdp_bool
        if (trace_bool):
            self.trace = neuron_bert.neuron_bert_trace(
                tau=tau, v_reset=None, timestep=timestep, wait=wait, start_time=self.starttime)
        else:
            self.trace = False
        self.trace_v = None
        if (self.convert):
            self.neuron_bert = neuron_bert.One_LIFNode_convert(tau=tau, v_reset=None, v_threshold=(
                2**(wait)*(1-(1-1/tau)/2)), timestep=timestep, wait=wait, start_time=self.starttime, biopolar_bool=False)

        else:
            self.neuron_bert = neuron_bert.One_LIFNode(tau=tau, v_reset=None, v_threshold=(
                tau**(wait)*(1-(1-1/tau)/2)), timestep=timestep, wait=wait, start_time=self.starttime, biopolar_bool=False)
        self.reset_threshold = (2**(wait)*(1-(1-1/tau)/2))

        self.tp = 0
        self.initialize = False
        self.tau = tau
        self.stdp_scale = 1
        self.x_sign = None
        self.x_shape = None
        self.md_shape = None

        self.spike_count_meter = AverageMeter()
        self.spike_counts = 0

        self.mem_count_meter = AverageMeter()
        self.mem_counts =0


    def forward(self, x):
        if (x is not None):

            # if (len(x.shape) != len(self.scale.shape)):
            #     if self.initialize == False:
            #         self.x_shape = x.size()
            #         batch_size = self.x_shape[0]//self.scale.shape[1]
            #         self.md_shape = [batch_size]+list(self.x_shape)
            #         self.md_shape[1] = self.md_shape[1]//batch_size
            #     x = x.view(self.md_shape)

            x = x/self.scale  # Note: scale(Mz) will be fused with weight
            if self.initialize == False:
                self.stdp_scale = 1
                self.spike_counts = 0
                self.mem_counts=0

                self.neuron_bert(torch.zeros_like(x))
                self.block = torch.zeros_like(x)
                self.x_sign = torch.zeros_like(x)
                self.trace_v = 0

                self.initialize = True

        if (self.t >= (self.starttime) and self.t < (self.starttime + self.timestep+self.wait)):
            if (self.stdp_bool and self.t == (self.starttime + self.wait)):
                if (self.x_shape is not None):
                    tp_md_shape = self.md_shape
                    tp_md_shape[-1] = 1
                    self.neuron_bert.v /= self.stdp_scale.view(tp_md_shape)
                else:
                    self.neuron_bert.v /= self.stdp_scale
            x = self.neuron_bert(x, time=self.t)

        else:
            x = None

        if (not (x == None)):
            x = torch.where(self.block == 1, 0, x)
            self.block = torch.where(x != 0, 1, self.block)

        if (self.trace and x is not None):
            trace_tp = self.trace(self.block, self.t)
            if (self.x_shape is not None):
                self.trace_pure = (trace_tp*(self.block)).view(self.x_shape)
                self.trace_v = (trace_tp*(self.block) *
                                self.scale).view(self.x_shape)
            else:
                self.trace_pure = (trace_tp*(self.block))
                self.trace_v = (trace_tp*(self.block)*self.scale)

        if (self.t == (self.starttime + self.timestep+self.wait)):
            self.reset(t_reset_bool=False)
        self.t += 1
        if (self.final_bool and self.t == (self.starttime + self.timestep)):
            return self.neuron_bert.v * self.scale

        if (x is not None):
            self.spike_counts += torch.count_nonzero(x)/(x.flatten()).size()[0]
            self.mem_counts += (1-self.block).sum()/(x.flatten()).size()[0]

            return x * self.scale
        else:
            return None

    def reset(self, t_reset_bool=True):
        if (t_reset_bool):
            self.t = 0
        self.block = 0
        self.tp = 0
        self.neuron_bert.reset()
        if (self.trace):
            self.trace.reset()
            self.trace_pure = 0
            self.trace_v = 0
        if (self.convert):
            self.neuron_bert.v_threshold = self.reset_threshold
        self.neuron_bert.v = 0.0
        self.x_sign = None
        self.stdp_scale = 1
        # if (self.spike_counts > 0):
        self.spike_count_meter.update(self.spike_counts)
        self.mem_count_meter.update(self.mem_counts)
        self.mem_counts=0
        self.spike_counts = 0
        self.initialize = False


class Scaledneuron_bert_onespike_time_bipolar(nn.Module):
    def __init__(self, scale=1., timestep=24, wait=12, start_time=0, i_layer=0, tau=2.0, convert=False, modulename=None, trace_bool=False, stdp_bool=False, scale_full=False, final_bool=False):
        super(Scaledneuron_bert_onespike_time_bipolar, self).__init__()
        self.scale_full = scale_full

        self.final_bool = final_bool

        self.scale = scale
        self.timestep = timestep
        self.wait = wait
        self.starttime = start_time
        self.t = 0
        self.convert = convert
        self.block = 0
        self.module_name = modulename
        self.i_layer = i_layer
        self.stdp_bool = stdp_bool
        if (trace_bool):
            self.trace = neuron_bert.neuron_bert_trace(
                tau=tau, v_reset=None, timestep=timestep, wait=wait, start_time=self.starttime)
        else:
            self.trace = False
        self.trace_v = None
        if (self.convert):
            self.neuron_bert = neuron_bert.One_LIFNode_convert(tau=tau, v_reset=None, v_threshold=(
                2**(wait)*(1-(1-1/tau)/2)), timestep=timestep, wait=wait, start_time=self.starttime, biopolar_bool=True)

        else:
            self.neuron_bert = neuron_bert.One_LIFNode(tau=tau, v_reset=None, v_threshold=(
                tau**(wait)*(1-(1-1/tau)/2)), timestep=timestep, wait=wait, start_time=self.starttime, biopolar_bool=True)
        self.reset_threshold = (2**(wait)*(1-(1-1/tau)/2))

        self.tp = 0
        self.initialize = False
        self.tau = tau
        self.stdp_scale = 1
        self.x_sign = None
        self.x_shape = None
        self.md_shape = None
        self.spike_count_meter = AverageMeter()
        self.spike_counts = 0
        self.mem_count_meter = AverageMeter()
        self.mem_counts =0
    def forward(self, x):

        if (x is not None):

            # if (len(x.shape) != len(self.scale.shape)):
            #     if self.initialize == False:
            #         self.x_shape = x.size()
            #         batch_size = self.x_shape[0]//self.scale.shape[1]
            #         self.md_shape = [batch_size]+list(self.x_shape)
            #         self.md_shape[1] = self.md_shape[1]//batch_size
            #     x = x.view(self.md_shape)

            x = x/self.scale  # / (2**24)
            if self.initialize == False:
                self.stdp_scale = 1
                self.spike_counts = 0

                self.neuron_bert(torch.zeros_like(x))
                self.block = torch.zeros_like(x)
                self.x_sign = torch.zeros_like(x)
                self.trace_v = 0
                self.mem_counts=0

                self.initialize = True

        if (self.t >= (self.starttime) and self.t < (self.starttime + self.timestep+self.wait)):
            if (self.stdp_bool and self.t == (self.starttime + self.wait)):
                if (self.x_shape is not None):
                    tp_md_shape = self.md_shape
                    tp_md_shape[-1] = 1
                    self.neuron_bert.v /= self.stdp_scale.view(tp_md_shape)
                else:
                    self.neuron_bert.v /= self.stdp_scale
            x = self.neuron_bert(x, time=self.t)

        else:
            x = None

        if (not (x == None)):
            x = torch.where(self.block == 1, 0, x)
            self.block = torch.where(x != 0, 1, self.block)

            self.x_sign = torch.where(x != 0, x, self.x_sign)



        if (self.trace and x is not None):
            trace_tp = self.trace(self.block, self.t)
            if (self.x_shape is not None):
                self.trace_pure = (
                    trace_tp*self.x_sign*(self.block)).view(self.x_shape)  # * self.scale
                self.trace_v = (trace_tp*self.x_sign*(self.block)
                                * self.scale).view(self.x_shape)  # * self.scale
            else:
                self.trace_pure = (trace_tp*self.x_sign *
                                   (self.block))  # * self.scale
                self.trace_v = (trace_tp*self.x_sign*(self.block)
                                * self.scale)  # * self.scale

        if (self.t == (self.starttime + self.timestep+self.wait)):
            self.reset(t_reset_bool=False)
        self.t += 1
        if (self.final_bool and self.t == (self.starttime + self.timestep)):
            return self.neuron_bert.v * self.scale



        if (x is not None):
            self.spike_counts += torch.count_nonzero(x)/(x.flatten()).size()[0]
            self.mem_counts += (1-self.block).sum()/(x.flatten()).size()[0]

            return x * self.scale
        else:
            return None

    def reset(self, t_reset_bool=True):
        if (t_reset_bool):
            self.t = 0
        self.block = 0
        self.tp = 0
        self.neuron_bert.reset()
        if (self.trace):
            self.trace.reset()
            self.trace_pure = 0
            self.trace_v = 0
        if (self.convert):
            self.neuron_bert.v_threshold = self.reset_threshold
        self.neuron_bert.v = 0.0
        self.x_sign = None
        self.stdp_scale = 1
        # if (self.spike_counts > 0):
        self.spike_count_meter.update(self.spike_counts)
        self.mem_count_meter.update(self.mem_counts)
        self.mem_counts=0
        self.spike_counts = 0
        self.initialize = False


class WTA_layer_neuron_bert(nn.Module):
    def __init__(self, scale=1., timestep=24, wait=12, start_time=0, i_layer=0, tau=2.0, convert=False, modulename=None, trace_bool=False):
        super(WTA_layer_neuron_bert, self).__init__()

        tp_scale = scale  # .unsqueeze(-1)
        self.scale = tp_scale
        self.timestep = timestep
        self.wait = wait
        self.starttime = start_time
        self.t = 0
        self.convert = convert
        self.block = 0
        self.module_name = modulename
        self.i_layer = i_layer
        self.log_base = 1.0/torch.log(torch.tensor(tau))
        if (trace_bool):
            self.trace = neuron_bert.neuron_bert_trace(
                tau=tau, v_reset=None, timestep=timestep, wait=wait, start_time=self.starttime)
        else:
            self.trace = False
        self.trace_v = None

        self.tp = 0
        self.initialize = False
        self.tau = tau
        self.wta_bool = True
        self.block_wta = 0
        self.accum_neuron_bert = neuron_bert.One_LIFNode(tau=tau, v_reset=None, v_threshold=(self.tau**(self.timestep-1)*(
            1-(1-1/tau)/2)), timestep=timestep, wait=0, start_time=self.starttime+wait, biopolar_bool=False)

        self.neuron_bert = neuron_bert.WTA_neuron_bert(tau=tau, v_reset=None, v_threshold=float((self.tau**(self.timestep-1))/self.log_base),
                                        timestep=timestep, wait=wait, start_time=self.starttime, threshold_mv=(self.tau**(self.timestep-1))/self.log_base)
        self.reset_threshold = (self.tau**(self.timestep-1))/self.log_base

        self.spike_sum = 0
        self.spike_sum_max = None
        self.x_sign = None
        self.spike_count_meter = AverageMeter()
        self.spike_counts = 0
        self.mem_count_meter = AverageMeter()
        self.mem_counts =0
    def forward(self, x):

        if (x is not None):
            if self.initialize == False:
                self.neuron_bert(torch.zeros_like(x))
                self.accum_neuron_bert(torch.zeros_like(x.sum(-1)))
                self.spike_counts = 0

                self.block = torch.zeros_like(x)
                self.block_input = torch.ones_like(x.max(-1).values)
                self.mem_counts=0

                self.trace_v = 0
                self.spike_sum = 0

                self.initialize = True

        if (self.t >= (self.starttime) and self.t < (self.starttime + self.timestep+self.wait)):
            if (x is not None):
                tp_max = x.max(-1).values

                x = x-x.max(-1).values.unsqueeze(-1) * \
                    self.block_input.unsqueeze(-1)
                self.block_input = torch.where(
                    tp_max > 0.0, 0, self.block_input)

            x = self.neuron_bert(x, time=self.t)

        else:
            x = None

        if (not (x == None)):
            x = torch.where(self.block == 1, 0, x)
            self.block = torch.where(x > 0, 1, self.block)

        if (self.trace and x is not None):
            # *self.x_sign #* self.scale
            self.trace_v = self.trace(self.block, self.t)*(self.block)

        if (self.t >= (self.starttime + self.wait) and self.t < (self.starttime + self.timestep+self.wait)):
            self.spike_sum *= self.tau
            self.spike_sum += x.sum(-1)/(self.tau**(self.timestep))

        if (self.t == (self.starttime + self.timestep+self.wait+3)):
            self.reset(t_reset_bool=False)

        self.t += 1

        if (x is not None):
            self.spike_counts += torch.count_nonzero(x)/(x.flatten()).size()[0]
            self.mem_counts += (1-self.block).sum()/(x.flatten()).size()[0]

            return x
        else:
            return None

    def reset(self, t_reset_bool=True):
        if (t_reset_bool):
            self.t = 0
        self.block = 0
        self.tp = 0
        self.spike_sum = 0
        if (self.trace):
            self.trace.reset()
            self.trace_pure = 0
            self.trace_v = 0
        self.block_input = 0
        self.neuron_bert.reset()
        self.accum_neuron_bert.reset()
        # if (self.spike_counts > 0):
        self.spike_count_meter.update(self.spike_counts)
        self.mem_count_meter.update(self.mem_counts)
        self.mem_counts=0
        self.spike_counts = 0
        self.neuron_bert.v_threshold = self.reset_threshold
        self.neuron_bert.v = 0.0
        self.initialize = False
        self.wta_bool = True


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class Scaledneuron_bert_onespike_time_double(nn.Module):
    def __init__(self, scale=1., timestep = 24, wait =12,start_time=0 , i_layer=0, tau = 2.0,convert = False, modulename = None,trace_bool = False, stdp_bool = False,scale_full =False,final_bool =False):
        super(Scaledneuron_bert_onespike_time_double, self).__init__()
        # print(modulename,start_time,i_layer)
        self.scale_full = scale_full
        if(self.scale_full):
            tp_scale = scale.unsqueeze(0)
        else:
            tp_scale = scale.unsqueeze(-1).unsqueeze(0)
        self.final_bool = final_bool
        # self.block_zero = torch.ones_like(tp_scale)
        # self.block_zero[tp_scale==0] = 0
        self.scale = tp_scale
        self.timestep = timestep
        self.wait = wait
        self.starttime = start_time
        self.t = 0
        self.convert = convert
        self.block =0
        self.module_name = modulename
        self.i_layer = i_layer
        self.stdp_bool = stdp_bool
        self.tau = tau
        self.log_base = 1.0/torch.log(torch.tensor(tau))
        if(trace_bool):
            self.trace = neuron_bert.neuron_bert_trace(tau=tau,v_reset=None,timestep = timestep,wait =wait, start_time= self.starttime)
        else:
            self.trace = False
        self.trace_v = None
        self.neuron_bert = neuron_bert.double_threshold_neuron_bert(tau=tau,v_reset=None, v_threshold=(tau**(wait))*(1-(1-1/tau)/2),timestep = timestep,wait =wait, start_time= self.starttime,biopolar_bool = False,threshold_shift=float((self.tau**(self.timestep-1))/self.log_base/2),threshold_mv=((self.tau**(self.timestep-1))/self.log_base),scale=self.scale)
        self.reset_threshold = (self.tau**(self.timestep-1))/self.log_base/2
        self.tp =0
        self.initialize = False
        self.stdp_scale = 1
        self.x_sign = None
        self.x_shape = None
        self.md_shape = None
        self.spike_sum =0
        self.spike_count_meter = AverageMeter()
        self.spike_counts =0
        self.FLOPS = 0 ## TODOlist
    def forward(self, x):
        if(x is not None):
            # if(self.i_layer==3):
            #     assert False, (x.shape, self.scale.shape)
            if(len(x.shape)!=len(self.scale.shape)):
                if self.initialize == False:
                    self.x_shape = x.size()
                    batch_size = self.x_shape[0]//self.scale.shape[1]
                    self.md_shape = [batch_size]+list(self.x_shape)
                    self.md_shape[1] =self.md_shape[1]//batch_size
                x= x.view(self.md_shape)
                # if(self.i_layer==2):
            x = x/self.scale  #/ (2**24)
            # assert(x.all()<=1 and x.all()>=-1)
            if self.initialize == False:
                self.stdp_scale =1
                self.spike_counts =0
                self.neuron_bert(torch.zeros_like(x) )
                self.block = torch.zeros_like(x)
                self.block_input = torch.ones_like(x.max(-1).values)
                self.x_sign = torch.zeros_like(x)
                self.trace_v =0
                self.initialize = True
                # if(self.trace):
                #     self.trace(torch.ones_like(x))
        if(self.t >= (self.starttime) and self.t < (self.starttime + 2*self.timestep+self.wait) ):
            if(self.t == (self.starttime + self.timestep+self.wait)):
                self.neuron_bert.v /= self.tau**self.timestep
            x= self.neuron_bert(x,time = self.t)
            if(self.t >= (self.starttime+self.wait) and self.t < (self.starttime + self.timestep+self.wait) ):
                if(x is not None):
                    tp_max = x.max(-1).values
                    self.neuron_bert.v -= tp_max.unsqueeze(-1)*self.block_input.unsqueeze(-1)*self.neuron_bert.v_threshold
                    self.block_input = torch.where(tp_max>0.0,0,self.block_input)
                    x = None
            # else:
            #     if(x is not None):
            #         print(x.sum())
        else:
            x = None
        if(not(x== None)):
            x = torch.where(self.block==1,0,x)
            self.block = torch.where(x>0,1,self.block)
        if(self.trace and x is not None):
            trace_tp = self.trace(self.block,self.t)
            if(self.x_shape is not None):
                self.trace_v = (trace_tp*(self.block)).view(self.x_shape) #* self.scale
            else:
                self.trace_v = (trace_tp*(self.block)) #* self.scale
        if(self.t >= (self.starttime +self.wait+ self.timestep) and self.t < (self.starttime + self.timestep*2+self.wait) ):
            self.spike_sum *= self.tau
            if(self.x_shape is not None):
                self.spike_sum +=x.view(self.x_shape).sum(-1)/(self.tau**(self.timestep-1))
            else:
                self.spike_sum +=x.sum(-1)/(self.tau**(self.timestep))
        if(self.t == (self.starttime + self.timestep*2+self.wait+1) ):
            self.reset(t_reset_bool=False)
        self.t += 1
        if(x is not None):
                self.spike_counts += torch.count_nonzero(x)/(x.flatten()).size()[0]
                if(self.x_shape is not None):
                    return (x ).view(self.x_shape)
                else:
                    return x
        else:
                return None
    def reset(self,t_reset_bool=True):
        if(t_reset_bool):
            self.t = 0
        self.block = 0
        self.tp =0
        self.neuron_bert.reset()
        if(self.trace):
            self.trace.reset()
            self.trace_pure =0
            self.trace_v =0
        # if(self.spike_counts>0):
        self.spike_count_meter.update(self.spike_counts)
        self.spike_counts =0
        self.neuron_bert.v =0.0
        self.x_sign = None
        self.stdp_scale =1
        self.spike_sum =0
        self.neuron_bert.threshold_shift = self.reset_threshold
        self.initialize = False