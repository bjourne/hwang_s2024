from torch import nn
import torch.nn.functional as F
import torch
from torch.autograd import Function
# from spikingjelly.clock_driven import neuron
from spikingjelly.activation_based import neuron

class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input




class ScaledNeuron(nn.Module):
    def __init__(self, scale=1.):
        super(ScaledNeuron, self).__init__()
        self.scale = scale
        self.t = 0
        self.neuron = neuron.IFNode(v_reset=None)

    def forward(self, x):
        x = x / self.scale
        if self.t == 0:
            self.neuron(torch.ones_like(x) * 0.5)
        x = self.neuron(x)
        self.t += 1
        return x * self.scale

    def reset(self):
        self.t = 0
        self.neuron.reset()

class ScaledNeuron_onespike(nn.Module):
    def __init__(self, scale=1.):
        super(ScaledNeuron_onespike, self).__init__()
        self.scale = scale
        self.t = 0
        self.block =0
        self.neuron = neuron.LIFNode(tau=-1/2,v_reset=None, v_threshold=(2.0**24)*(3/4))

    def forward(self, x):

        x = x / self.scale #/ (2**24)

        if self.t == 0:
            self.neuron(torch.zeros_like(x) )
            self.block = torch.zeros_like(x)

        x = self.neuron(x)


        # if self.t >20:
        

        x = x-self.block
        self.block = torch.where(x>0,1,self.block)
        # assert torch.all(x==0), x.sum()

        self.t += 1
        return x * self.scale

    def reset(self):
        self.t = 0
        self.block = 0

        self.neuron.reset()

class ScaledNeuron_onespike_time(nn.Module):
    def __init__(self, scale=1., timestep = 24, wait =12,start_time=0 , i_layer=0, tau = 2.0,convert = False, modulename = None,trace_bool = False):
        super(ScaledNeuron_onespike_time, self).__init__()
        tp_scale = scale.unsqueeze(0)#.unsqueeze(-1)
        self.block_zero = torch.ones_like(tp_scale)
        self.block_zero[tp_scale<=0] = 0
        self.scale = tp_scale
        self.timestep = timestep
        self.wait = wait
        self.starttime = start_time
        self.t = 0
        self.convert = convert
        self.block =0
        self.module_name = modulename
        self.reset_threshold = tau**(wait)
        self.i_layer = i_layer
        if(trace_bool):
            self.trace = neuron.neuron_trace(tau=tau,v_reset=None,timestep = timestep,wait =wait, start_time= self.starttime)
        else:
            self.trace = False
        self.trace_v = None
        if(self.convert):
            self.neuron = neuron.One_LIFNode_convert(tau=tau,v_reset=None, v_threshold=(2**(wait)*(1-(1-1/tau)/2)),timestep = timestep,wait =wait, start_time= self.starttime)
            self.reset_threshold = 2**wait
        else:
            self.neuron = neuron.One_LIFNode(tau=tau,v_reset=None, v_threshold=(tau**(wait)*(1-(1-1/tau)/2)),timestep = timestep,wait =wait, start_time= self.starttime)

        self.tp =0
        self.initialize = False
        self.tau = tau
    def forward(self, x):

        if(x is not None):

            x = x*self.block_zero/self.scale  #/ (2**24)
            assert(x.all()<=1)
            if self.initialize == False:
                self.neuron(torch.zeros_like(x) )
                self.block = torch.zeros_like(x)
                self.initialize = True
                if(self.trace):
                    self.trace(torch.zeros_like(x) )

        if(self.t >= (self.starttime) and self.t < (self.starttime + self.timestep+self.wait) ):
            x,self.v = self.neuron(x,time = self.t)
        else:
            x = None

        if(not(x== None)):
            x = torch.where(self.block==1,0,x)
            self.block = torch.where(x>0,1,self.block)

        if(self.t >= (self.starttime +self.wait) and self.t < (self.starttime + self.timestep+self.wait) ):

            self.tp *=self.tau
            self.tp +=x
        
        
        if(self.trace and x is not None):
            self.trace_v = self.trace(x,self.t)*self.block_zero * self.scale
       
        self.t += 1

        # if(self.starttime==0 and  x.max()>0): #self.t == (self.starttime+4)):
        #         assert False, (x.max(),self.t)

        

        # if(self.i_layer==33 and self.t == (self.starttime + self.timestep+self.wait)):

        #      import numpy as np
        #      np.savetxt('snn.txt', (self.tp*self.scale).cpu().flatten().numpy())
        #      assert False, self.scale

        # if(self.trace):
        #     if(self.trace_v is not None):
        #         return self.trace_v*self.block_zero*self.scale
        #     else:
        #         return None
        # else:
        if(x is not None):
                return x*self.block_zero * self.scale
        else:
                return None
    def reset(self):

        self.tp =0
        if(self.trace):
                    self.trace(torch.zeros_like(x) )
        self.t = 0
        self.block = 0
        self.tp =0
        if(self.convert):
            self.neuron.v_threshold = self.reset_threshold
        self.neuron.reset()
        self.initialize = False


class ScaledNeuron_onespike_time_relu2(nn.Module):
    def __init__(self, scale=1., timestep = 24, wait =12,start_time=0 , i_layer=0, tau = 2.0,convert = False, modulename = None,trace_bool = False, stdp_bool = False,scale_full =False,final_bool =False):
        super(ScaledNeuron_onespike_time_relu2, self).__init__()
        self.scale_full = scale_full
        if(self.scale_full):
            tp_scale = scale.unsqueeze(0)

        else:    
            tp_scale = scale.unsqueeze(-1).unsqueeze(0)
        # print(tp_scale.shape)
        # assert tp_scale.all()>0, "scale is strange"
        self.block_zero = torch.ones_like(tp_scale)
        self.block_zero[tp_scale<=0] = 0
        self.final_bool = final_bool

        self.scale = tp_scale
        self.timestep = timestep
        self.wait = wait
        self.starttime = start_time
        self.t = 0
        self.convert = convert
        self.block =0
        self.module_name = modulename
        # self.reset_threshold = tau**(wait)
        self.i_layer = i_layer
        self.stdp_bool = stdp_bool
        if(trace_bool):
            self.trace = neuron.neuron_trace(tau=tau,v_reset=None,timestep = timestep,wait =wait, start_time= self.starttime)
        else:
            self.trace = False
        self.trace_v = None
        if(self.convert):
            self.neuron = neuron.One_LIFNode_convert(tau=tau,v_reset=None, v_threshold=(2**(wait)*(1-(1-1/tau)/2)),timestep = timestep,wait =wait, start_time= self.starttime,biopolar_bool = False)
        else:
            self.neuron = neuron.One_LIFNode(tau=tau,v_reset=None, v_threshold=(tau**(wait)*(1-(1-1/tau)/2)),timestep = timestep,wait =wait, start_time= self.starttime,biopolar_bool = False)
        self.reset_threshold = (2**(wait)*(1-(1-1/tau)/2))

        self.tp =0
        self.initialize = False
        self.tau = tau
        self.stdp_scale = 1

        self.x_sign = None
        self.x_shape = None
        self.md_shape = None

    def forward(self, x):

        if(x is not None):
            if(len(x.shape)!=len(self.scale.shape)):
                if self.initialize == False:
                    self.x_shape = x.size()
                    batch_size = self.x_shape[0]//self.scale.shape[1]
                    self.md_shape = [batch_size]+list(self.x_shape)
                    self.md_shape[1] =self.md_shape[1]//batch_size
                x= x.view(self.md_shape)

            x = x/self.scale  
            # assert(x.all()<=1 and x.all()>=-1)
            if self.initialize == False:
                self.stdp_scale =1
                self.trace_v =0
                self.neuron(torch.zeros_like(x) )
                self.block = torch.zeros_like(x)
                self.initialize = True
                if(self.trace):
                    self.trace(torch.ones_like(x)/self.tau )

        if(self.t >= (self.starttime) and self.t < (self.starttime + self.timestep+self.wait) ):
            # if(self.stdp_bool and self.t == (self.starttime +self.wait)):
            #     self.neuron.v/=self.stdp_scale
            
            if(self.stdp_bool and self.t == (self.starttime +self.wait)):
                # assert False, (self.neuron.v.shape,self.stdp_scale.shape)
                if(self.x_shape is not None):
                    tp_md_shape = self.md_shape
                    tp_md_shape[-1] =1
                    self.neuron.v/=self.stdp_scale.view(tp_md_shape)
                else:
                    self.neuron.v/=self.stdp_scale
            
            x= self.neuron(x,time = self.t)
            # if(x is not None):
            #     assert(x.all()>=0)

        else:
            x = None
        
        if(not(x== None)):
            x = torch.where(self.block==1,0,x)
            # assert x.all()>=0, "negative spike on ReLU layer"
            self.block = torch.where(x>0,1,self.block)
            x = x*self.block_zero

        if(self.t >= (self.starttime +self.wait) and self.t < (self.starttime + self.timestep+self.wait) ):

            self.tp *=self.tau
            self.tp +=x
        # if(self.trace and x is not None):
        #     #self.trace_v = self.trace(x,self.t) * self.scale
        #     self.trace_v = self.trace(self.block,self.t)*(self.block)* self.scale #* self.scale
        
        if(self.trace and x is not None):
            trace_tp = self.trace(self.block,self.t)
            if(self.x_shape is not None):
                self.trace_pure = (trace_tp*(self.block)).view(self.x_shape) #* self.scale
                self.trace_v = (trace_tp*(self.block)*self.scale).view(self.x_shape) #* self.scale
            else:
                self.trace_pure = (trace_tp*(self.block)) #* self.scale
                self.trace_v = (trace_tp*(self.block)*self.scale) #* self.scale 
                
        
        
        if(self.t == (self.starttime + self.timestep+self.wait) ):
            self.reset(t_reset_bool= False)
        
        
        self.t += 1

        if(self.final_bool and self.t == (self.starttime + self.timestep)):
            return self.neuron.v * self.scale

        # if( self.i_layer==66 and self.t == (self.starttime + self.timestep+self.wait)):
        #      import numpy as np
        #      tp_softmax = nn.Softmax(dim=-1)

        #      np.savetxt('snn.txt', ((self.tp*self.scale/(self.tau**(self.timestep-1)))[0].cpu().flatten().numpy()))
        #      assert False, (self.module_name,self.scale.shape)

        # if(x is not None):
        #         return x * self.scale
        # else:
        #         return None

        if(x is not None):
                if(self.x_shape is not None):

                    return (x * self.scale).view(self.x_shape)

                else:
                    return x * self.scale
        else:
                return None
    def reset(self,t_reset_bool=True):
        if(t_reset_bool):
            self.t = 0
        self.block = 0
        self.tp =0
        self.neuron.reset()
        if(self.trace):
            self.trace.reset()
            self.trace_v =0
            self.trace_pure =0

        if(self.convert):
            self.neuron.v_threshold = self.reset_threshold
        self.neuron.v =0.0
        self.stdp_scale =1

        self.initialize = False

class ScaledNeuron_onespike_time_relu(nn.Module):
    def __init__(self, scale=1., timestep = 24, wait =12,start_time=0 , i_layer=0, tau = 2.0,convert = False, modulename = None,trace_bool = False, stdp_bool = False,scale_full =False,final_bool =False):
        super(ScaledNeuron_onespike_time_relu, self).__init__()
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
        if(trace_bool):
            self.trace = neuron.neuron_trace(tau=tau,v_reset=None,timestep = timestep,wait =wait, start_time= self.starttime)
        else:
            self.trace = False
        self.trace_v = None
        if(self.convert):
            self.neuron = neuron.One_LIFNode_convert(tau=tau,v_reset=None, v_threshold=(2**(wait)*(1-(1-1/tau)/2)),timestep = timestep,wait =wait, start_time= self.starttime,biopolar_bool = False)
            # self.reset_threshold = 2**wait
        # elif(self.stdp_bool):
        #     self.neuron = neuron.One_LIFNode(tau=tau,v_reset=None, v_threshold=(tau**(wait)),timestep = timestep,wait =wait, start_time= self.starttime,biopolar_bool = True)

        else:
            self.neuron = neuron.One_LIFNode(tau=tau,v_reset=None, v_threshold=(tau**(wait)*(1-(1-1/tau)/2)),timestep = timestep,wait =wait, start_time= self.starttime,biopolar_bool = False)
        self.reset_threshold = (2**(wait)*(1-(1-1/tau)/2))

        self.tp =0
        self.initialize = False
        self.tau = tau
        self.stdp_scale = 1
        self.x_sign = None
        self.x_shape = None
        self.md_shape = None
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

                self.neuron(torch.zeros_like(x) )
                self.block = torch.zeros_like(x)                
                self.x_sign = torch.zeros_like(x)
                self.trace_v =0

                self.initialize = True
                if(self.trace):
                    self.trace(torch.zeros_like(x) )

        if(self.t >= (self.starttime) and self.t < (self.starttime + self.timestep+self.wait) ):
            if(self.stdp_bool and self.t == (self.starttime +self.wait)):
                # assert False, (self.neuron.v.shape,self.stdp_scale.shape)
                if(self.x_shape is not None):
                    tp_md_shape = self.md_shape
                    tp_md_shape[-1] =1
                    self.neuron.v/=self.stdp_scale.view(tp_md_shape)
                else:
                    self.neuron.v/=self.stdp_scale
            x= self.neuron(x,time = self.t)
            # if(x is not None):
            #     assert(x.all()>=0)

        else:
            x = None

        if(not(x== None)):
            x = torch.where(self.block==1,0,x)
            self.block = torch.where(x!=0,1,self.block)

            self.x_sign = torch.where(x!=0,x,self.x_sign)

        # if(self.t >= (self.starttime +self.wait) and self.t < (self.starttime + self.timestep+self.wait) ):

        #     self.tp *=self.tau
        #     self.tp +=x

        # if(self.t >= (self.starttime +self.wait) and self.t < (self.starttime + self.timestep+self.wait) ):

        #     self.tp *=self.tau
        #     self.tp +=x

        if(self.trace and x is not None):
            trace_tp = self.trace(self.block,self.t)
            if(self.x_shape is not None):
                self.trace_pure = (trace_tp*self.x_sign*(self.block)).view(self.x_shape) #* self.scale
                self.trace_v = (trace_tp*self.x_sign*(self.block)*self.scale).view(self.x_shape) #* self.scale
            else:
                self.trace_pure = (trace_tp*self.x_sign*(self.block)) #* self.scale
                self.trace_v = (trace_tp*self.x_sign*(self.block)*self.scale) #* self.scale 
        
        
        # if(x is None and self.t == (self.starttime + self.timestep+self.wait) ):
        #     self.reset()


        if(self.t == (self.starttime + self.timestep+self.wait) ):
            self.reset(t_reset_bool=False)  
        self.t += 1
        if(self.final_bool and self.t == (self.starttime + self.timestep)):
            return self.neuron.v * self.scale


        # if( self.i_layer==16 and self.t == (self.starttime + self.timestep+self.wait)):
        #      import numpy as np
        #      tp_softmax = nn.Softmax(dim=-1)

        #      np.savetxt('snn.txt', ((self.tp*self.scale/(self.tau**(self.timestep-1)))[0].cpu().flatten().numpy()))
        
        #     #  np.savetxt('snn.txt', ((self.trace_v))[0].cpu().flatten().numpy())

        #     #  np.savetxt('snn.txt', ((self.v*self.scale/(self.tau**(self.timestep-1)))[0].cpu().flatten().numpy()))
             
        #      assert False, (self.x_shape,self.module_name,self.tp.shape,self.scale.shape)
        
        # if( self.i_layer==66 and self.t == (self.starttime + self.timestep+self.wait)):
        #      import numpy as np
        #      tp_softmax = nn.Softmax(dim=-1)

        #      np.savetxt('snn.txt', ((self.tp*self.scale/(self.tau**(self.timestep-1)))[0].cpu().flatten().numpy()))
        #      assert False, (self.module_name,self.scale.shape)
        if(x is not None):
                if(self.x_shape is not None):

                    return (x * self.scale).view(self.x_shape)

                else:
                    return x * self.scale
        else:
                return None
    def reset(self,t_reset_bool=True):
        if(t_reset_bool):
            self.t = 0
        self.block = 0
        self.tp =0
        self.neuron.reset()
        if(self.trace):
            self.trace.reset()
            self.trace_pure =0
            self.trace_v =0
        if(self.convert):
            self.neuron.v_threshold = self.reset_threshold
        self.neuron.v =0.0
        self.x_sign = None
        self.stdp_scale =1

        self.initialize = False





class ScaledNeuron_onespike_time_bipolar(nn.Module):
    def __init__(self, scale=1., timestep = 24, wait =12,start_time=0 , i_layer=0, tau = 2.0,convert = False, modulename = None,trace_bool = False, stdp_bool = False,scale_full =False,final_bool =False):
        super(ScaledNeuron_onespike_time_bipolar, self).__init__()
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
        if(trace_bool):
            self.trace = neuron.neuron_trace(tau=tau,v_reset=None,timestep = timestep,wait =wait, start_time= self.starttime)
        else:
            self.trace = False
        self.trace_v = None
        if(self.convert):
            self.neuron = neuron.One_LIFNode_convert(tau=tau,v_reset=None, v_threshold=(2**(wait)*(1-(1-1/tau)/2)),timestep = timestep,wait =wait, start_time= self.starttime,biopolar_bool = True)
            # self.reset_threshold = 2**wait
        # elif(self.stdp_bool):
        #     self.neuron = neuron.One_LIFNode(tau=tau,v_reset=None, v_threshold=(tau**(wait)),timestep = timestep,wait =wait, start_time= self.starttime,biopolar_bool = True)

        else:
            self.neuron = neuron.One_LIFNode(tau=tau,v_reset=None, v_threshold=(tau**(wait)*(1-(1-1/tau)/2)),timestep = timestep,wait =wait, start_time= self.starttime,biopolar_bool = True)
        self.reset_threshold = (2**(wait)*(1-(1-1/tau)/2))

        self.tp =0
        self.initialize = False
        self.tau = tau
        self.stdp_scale = 1
        self.x_sign = None
        self.x_shape = None
        self.md_shape = None
    def forward(self, x):
        # if( self.i_layer==128 ):
        #     assert False, "it work"
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

                self.neuron(torch.zeros_like(x) )
                self.block = torch.zeros_like(x)                
                self.x_sign = torch.zeros_like(x)
                self.trace_v =0

                self.initialize = True
                if(self.trace):
                    self.trace(torch.zeros_like(x) )

        if(self.t >= (self.starttime) and self.t < (self.starttime + self.timestep+self.wait) ):
            if(self.stdp_bool and self.t == (self.starttime +self.wait)):
                # assert False, (self.neuron.v.shape,self.stdp_scale.shape)
                if(self.x_shape is not None):
                    tp_md_shape = self.md_shape
                    tp_md_shape[-1] =1
                    self.neuron.v/=self.stdp_scale.view(tp_md_shape)
                else:
                    self.neuron.v/=self.stdp_scale
            x= self.neuron(x,time = self.t)
            # if(x is not None):
            #     assert(x.all()>=0)

        else:
            x = None

        if(not(x== None)):
            x = torch.where(self.block==1,0,x)
            self.block = torch.where(x!=0,1,self.block)

            self.x_sign = torch.where(x!=0,x,self.x_sign)


        # if(self.t >= (self.starttime +self.wait) and self.t < (self.starttime + self.timestep+self.wait) ):

        #     self.tp *=self.tau
        #     # if(self.x_shape is not None):
        #     #     self.tp +=x.view(self.x_shape)
        #     # else:
        #     self.tp +=x

        if(self.trace and x is not None):
            trace_tp = self.trace(self.block,self.t)
            if(self.x_shape is not None):
                self.trace_pure = (trace_tp*self.x_sign*(self.block)).view(self.x_shape) #* self.scale
                self.trace_v = (trace_tp*self.x_sign*(self.block)*self.scale).view(self.x_shape) #* self.scale
            else:
                self.trace_pure = (trace_tp*self.x_sign*(self.block)) #* self.scale
                self.trace_v = (trace_tp*self.x_sign*(self.block)*self.scale) #* self.scale 
        
        
        # if(x is None and self.t == (self.starttime + self.timestep+self.wait) ):
        #     self.reset()


        if(self.t == (self.starttime + self.timestep+self.wait) ):
            self.reset(t_reset_bool=False)  
        self.t += 1
        if(self.final_bool and self.t == (self.starttime + self.timestep)):
            return self.neuron.v * self.scale


        if( self.i_layer==128 and self.t == (self.starttime + self.timestep+self.wait)):
             import numpy as np
             tp_softmax = nn.Softmax(dim=-1)
             np.savetxt('snn.txt', ((self.tp*self.scale/(self.tau**(self.timestep-1)))[0].cpu().flatten().numpy()))
        
            #  np.savetxt('snn.txt', ((self.trace_v))[0].cpu().flatten().numpy())

            #  np.savetxt('snn.txt', ((self.v*self.scale/(self.tau**(self.timestep-1)))[0].cpu().flatten().numpy()))
             
             assert False, (self.x_shape,self.module_name,self.tp.shape,self.scale.shape)
        
        if(x is not None):
                if(self.x_shape is not None):

                    return (x * self.scale).view(self.x_shape)

                else:
                    return x * self.scale
        else:
                return None
    def reset(self,t_reset_bool=True):
        if(t_reset_bool):
            self.t = 0
        self.block = 0
        self.tp =0
        self.neuron.reset()
        if(self.trace):
            self.trace.reset()
            self.trace_pure =0
            self.trace_v =0
        if(self.convert):
            self.neuron.v_threshold = self.reset_threshold
        self.neuron.v =0.0
        self.x_sign = None
        self.stdp_scale =1

        self.initialize = False



class WTA_layer_Neuron(nn.Module):
    def __init__(self, scale=1., timestep = 24, wait =12,start_time=0 , i_layer=0, tau = 2.0,convert = False, modulename = None,trace_bool = False):
        super(WTA_layer_Neuron, self).__init__()
        tp_scale = scale#.unsqueeze(-1)
        # self.block_zero = torch.ones_like(tp_scale)
        # self.block_zero[tp_scale<=0] = 0
        self.scale = tp_scale
        self.timestep = timestep
        self.wait = wait
        self.starttime = start_time
        self.t = 0
        self.convert = convert
        self.block =0
        self.module_name = modulename
        self.i_layer = i_layer
        self.log_base = 1.0/torch.log(torch.tensor(tau))
        if(trace_bool):
            self.trace = neuron.neuron_trace(tau=tau,v_reset=None,timestep = timestep,wait =wait, start_time= self.starttime)
        else:
            self.trace = False
        self.trace_v = None


        self.tp =0
        self.initialize = False
        self.tau = tau
        self.wta_bool = True
        self.block_wta =0
        #self.neuron = neuron.WTA_neuron(tau=tau,v_reset=None, v_threshold=(0.0),timestep = timestep,wait =wait, start_time= self.starttime , threshold_mv = (self.tau**(self.timestep-1))/self.log_base)
        self.neuron = neuron.WTA_neuron(tau=tau,v_reset=None, v_threshold=float((self.tau**(self.timestep-1))/self.log_base/2),timestep = timestep,wait =wait, start_time= self.starttime , threshold_mv = (self.tau**(self.timestep-1))/self.log_base)
        self.reset_threshold = (self.tau**(self.timestep-1))/self.log_base/2

        self.spike_sum = 0
        self.spike_sum_max = None
        self.x_sign = None

        # self.block_input = None
    def forward(self, x):

        # assert False, (x.shape,self.scale.shape)
        if(x is not None):
            if self.initialize == False:
                self.neuron(torch.zeros_like(x) )
                self.block = torch.zeros_like(x)
                self.block_input = torch.ones_like(x.max(-1).values)

                self.trace_v =0
                self.spike_sum = 0

                self.initialize = True
                if(self.trace):
                    self.trace(torch.ones_like(x)/self.tau )

        if(self.t >= (self.starttime) and self.t < (self.starttime + self.timestep+self.wait) ):
            # print(x[0][0][0],self.t,(x[0][0][0]).sum())
            if(x is not None):
                tp_max = x.max(-1).values

                x =x-x.max(-1).values.unsqueeze(-1)*self.block_input.unsqueeze(-1)
                self.block_input = torch.where(tp_max>0.0,0,self.block_input)
            x= self.neuron(x,time = self.t)


        else:
            x = None

        if(not(x== None)):
            x = torch.where(self.block==1,0,x)
            self.block = torch.where(x>0,1,self.block)

        if(self.trace and x is not None):
            self.trace_v = self.trace(self.block,self.t)*(self.block)#*self.x_sign #* self.scale


        if(self.t >= (self.starttime +self.wait) and self.t < (self.starttime + self.timestep+self.wait) ):
            self.spike_sum *= self.tau

            self.spike_sum +=x.sum(-1)/(self.tau**(self.timestep-1))
        
        # if(self.t >= (self.starttime +self.wait) and self.t < (self.starttime + self.timestep+self.wait) ):

        #     self.tp *=self.tau
        #     self.tp +=x

        # if( self.i_layer==4 and self.t == (self.starttime + self.timestep+self.wait)):
        #      import numpy as np
        #      tp_softmax = nn.Softmax(dim=-1)
        #      #np.savetxt('snn.txt', ((self.tp*self.scale/(self.tau**(self.timestep-1)))[0][0].cpu().flatten().numpy()))
        #      np.savetxt('snn.txt', (self.tp/(self.tau**(self.timestep-1))/self.spike_sum.unsqueeze(-1))[0][0].cpu().flatten().numpy())

        #     #  np.savetxt('snn.txt', ((self.trace_v))[0].cpu().flatten().numpy())

        #     #  np.savetxt('snn.txt', ((self.v*self.scale/(self.tau**(self.timestep-1)))[0].cpu().flatten().numpy()))
             
        #      assert False, (self.module_name,self.tp.shape)


        if(self.t == (self.starttime + self.timestep+self.wait+3) ):
            self.reset(t_reset_bool=False)

        self.t += 1



        if(x is not None):
                # tp_shape = list(x.shape)
                # tp_shape[1] = int(tp_shape[1]*tp_shape[0])
                # tp_shape.pop(0)
                # x= x.view(tp_shape)
                return x #* self.scale
        else:
                return None
    def reset(self,t_reset_bool=True):
        if(t_reset_bool):
            self.t = 0
        self.block = 0
        self.tp =0
        self.spike_sum =0
        #if(self.convert):

        if(self.trace):
            self.trace.reset()
            self.trace_pure =0
            self.trace_v =0
        self.block_input =0
        self.neuron.reset()
        self.neuron.v_threshold = self.reset_threshold
        self.neuron.v =0.0
        self.initialize = False
        self.wta_bool = True






class MCN(nn.Module):
    def __init__(self, up=2., t=32):
        super().__init__()
        self.max_act = 0
        self.t = t
        self.batch_size =64
        self.relu_bool = False
        self.i_layer =0
        self.relu = nn.ReLU()
        self.name = None
        self.full_scale = False
        self.batch_fusion_bool = False

    def forward(self, x):

        if(self.relu_bool):
            x = self.relu(x)

        # if( self.i_layer==15):
        #     import numpy as np
        #     np.savetxt('ann.txt', (x)[0].cpu().flatten().numpy())
        #     assert False, self.name
        ######
        #Code for get distribution of Activation on Transformer
        ######
        
        ## find the scale of each neuron on inference
        if(isinstance(self.max_act,int)):
            if(x.shape[0]!=self.batch_size ):
                self.batch_fusion_bool = True
                tp_x = x.clone()
                tp_shape = list(x.size())
                tp_shape[0]= int(tp_shape[0]/self.batch_size)
                tp_shape = list([self.batch_size])+tp_shape                
                tp_x = tp_x.view(tp_shape)
                if(len(tp_shape)<5):
                    self.max_act = tp_x.abs().max(0).values
                    self.full_scale =True
                else:
                    self.max_act = tp_x.abs().max(0).values.max(-1).values
            else:
                if(len(x.shape)<4):
                    self.max_act = x.abs().max(0).values
                    self.full_scale =True
                else:
                    self.max_act = x.abs().max(0).values.max(-1).values
        else:
            if(self.batch_fusion_bool):
                tp_x = x.clone()
                tp_shape = list(x.shape)
                batch_size_tp = x.shape[0]//self.max_act.shape[0]
                tp_shape[0]= int(tp_shape[0]/batch_size_tp)
                tp_shape = list([batch_size_tp])+tp_shape                
                tp_x = tp_x.view(tp_shape)
                if(self.full_scale):
                    self.max_act =torch.maximum(tp_x.abs().max(0).values,self.max_act)
                else:
                    self.max_act =torch.maximum(tp_x.abs().max(0).values.max(-1).values,self.max_act)

            else:

        
                if(self.full_scale):
                    self.max_act =torch.maximum(x.abs().max(0).values,self.max_act)

                else:
                    self.max_act =torch.maximum(x.abs().max(0).values.max(-1).values,self.max_act)

        return x





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
