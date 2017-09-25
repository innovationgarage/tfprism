# From https://stackoverflow.com/a/44216268
# Hacked for non-gpu devices by egil@innovationgarage

from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.merge import Concatenate
import keras.layers
import tensorflow as tf
import contextlib
import copy
import types

def patch(origfn):
    def wrapper(newfn):
        def wrapper(*arg, **kw):
            return newfn(origfn, *arg, **kw)
        wrapper.func_name = origfn.func_name
        wrapper.__name__ = origfn.__name__
        return wrapper
    return wrapper

def format_node(node, indent = '', depth=None, done=None):
    if depth == 0:
        return '%s...\n' % indent
    if depth is not None:
        depth -= 1
    if done is None:
        done = set()
    if id(node) in done:
        return "%sAllready printed %s\n" % (indent, node.name)
    done.add(id(node))
    if isinstance(node, tf.Tensor):
        tensor = "%s%s: %s" % (node.dtype.name, node.shape, node.name)
        if node.device:
            tensor += " <%s>" % node.device
        return "%s%s\n%s" % (indent, tensor, format_node(node.op, indent + '  ', depth, done))
    elif isinstance(node, tf.Operation):
        op = "%s: %s" % (node.type, node.name)
        if node.device:
            op += " <%s>" % node.device
        control_input = ''
        input = ''
        if node.control_inputs:
            control_input = "%s  REQUIRE\n%s" % (
                indent,
                ''.join(format_node(input, indent + '    ', depth, done) for input in node.control_inputs
                        if input not in node.inputs))
        if node.inputs:
            input = "%s  <--\n%s" % (
                indent,
                ''.join(format_node(input, indent + '    ', depth, done) for input in node.inputs))
        output = ''
        if node.outputs:
            output = "%s  --> %s\n" % (
                indent,
                ', '.join(output.name for output in node.outputs))
        return "%s%s\n%s%s%s" % (
            indent,
            op,
            control_input,
            input,
            output
        )

def _is_merged_gradient(node, copier, dependencies, **kw):
    for category in ('inputs', 'control_inputs'):
        for item in dependencies[category]:
            if hasattr(item, '_merged_gradient'):
                return True
    return False

def merge_gradients(node, copier, purpose=None, **kw):
    if not isinstance(node, tf.Operation):
        return None

    elif 'gradients_' in node.name:
        if purpose == 'original_gradient':
            return None
        else:
            if copier.multiple.node_copiers.index(copier) != 0:
                return copier.multiple.node_copiers[0].copy(node)
            else:
                res = tf.reduce_mean(
                    tf.concat(
                        axis=0,
                        values=[other_copier.copy(node, 'original_gradient')
                                for other_copier in copier.multiple.node_copiers]),
                    0)
                res._merged_gradient = True
                return res

    elif _is_merged_gradient(node=node, copier=copier, purpose=purpose, **kw):
        if copier.multiple.node_copiers.index(copier) != 0:
            return copier.multiple.node_copiers[0].copy(node)
        else:
            res = copier.apply_next_filter(node=node, copier=copier, purpose=purpose, **kw)
            res._merged_gradient = True
            return res

    else:
        return None
        
def filter_variables(node, copier, **kw):
    if isinstance(node, tf.Operation) and node.type == 'VariableV2':
        return  node
    elif (    isinstance(node, tf.Operation)
          and node.type == 'Identity'
          and len(node.inputs) == 1
          and node.inputs[0].op.type == 'VariableV2'):
        return node
    else:
        return None

def set_device(device):
    def set_device(copier, **kw):
        with tf.device(device):
            return copier.apply_next_filter(copier=copier, **kw)
    return set_device

class SingleNodeCopier(object):
    def __init__(self, filters = (), prefix = 'copy_'):
        self.filters = filters + (self._copy,)
        self.prefix = prefix
        self.mapping = {}

    @classmethod
    def copy_node(cls, node, *arg, **kw):
        self = cls(*arg, **kw)
        res = self.copy(node)
        res._prism_copier = self
        return res

    def __getitem__(self, node):
        return self.copy(node)

    def __contains__(self, node):
        return id(node) in self.mapping

    def apply_next_filter(self, node, filters, **kw):
        res = None
        while res is None:
            filter = filters[0]
            filters = filters[1:]
            res = filter(node=node, filters=filters, **kw)
        return res
    
    def copy(self, node, purpose=None, **kw):
        key = id(node)
        if purpose is not None:
            key = (key, purpose)
        if key in self.mapping:
            return self.mapping[key]
        res = self.apply_next_filter(
            node=node,
            dependencies=self.copy_dependencies(node=node, purpose=purpose, **kw),
            filters=self.filters,
            copier=self,
            purpose=purpose,
            **kw)
        self.mapping[key] = res
        return res

    def _copy(self, node, **kw):
        if isinstance(node, tf.Tensor):
            return self.copy_tensor(node=node, **kw)
        elif isinstance(node, tf.Operation):
            return self.copy_op(node=node, **kw)
        else:
            raise ValueError("Unknown node type %s: %s" % (type(node), node))

    def copy_dependencies(self, node, purpose=None, **kw):
        if isinstance(node, tf.Tensor):
            return self.copy_tensor_dependencies(node=node, purpose=purpose, **kw)
        elif isinstance(node, tf.Operation):
            return self.copy_op_dependencies(node=node, purpose=purpose, **kw)
        else:
            raise ValueError("Unknown node type %s: %s" % (type(node), node))
    
    def copy_tensor_dependencies(self, node, **kw):
        return {
            'op': self.copy(node.op)
        }

    def copy_op_inputs(self, node):
        return [self.copy(input) for input in node.inputs]

    def copy_op_control_inputs(self, node):
        return [self.copy(input) for input in node.control_inputs]

    def copy_op_output_dtypes(self, node):
        return [to.dtype for to in node.outputs]

    def copy_op_attrs(self, node):
        attrs = dict(node.node_def.attr)
        if '_class' in attrs:
            attrs['_class'] = tf.AttrValue(list=tf.AttrValue.ListValue(s=[
                name.replace("@", "@%s" % self.prefix) for name in node.get_attr('_class')]))
        return attrs

    def copy_op_dependencies(self, node, **kw):
        return {
            'op_type': node.type,
            'control_inputs': self.copy_op_control_inputs(node),
            'inputs': self.copy_op_inputs(node),
            'dtypes': self.copy_op_output_dtypes(node),
            'name': self.prefix + node.name,
            'attrs': self.copy_op_attrs(node),
        }

    def create_op(self, control_inputs, **kw):
        with tf.control_dependencies(control_inputs):
            return tf.get_default_graph().create_op(
                compute_shapes=True,
                compute_device=True,
                **kw
            )
        
    def copy_op(self, node, dependencies, **kw):
        return self.create_op(
            **dependencies)
    
    def copy_tensor(self, node, dependencies, **kw):
        idx = node.op.outputs.index(node)
        return dependencies['op'].outputs[idx]

class MultipleNodeCopier(object):
    def __init__(self, *node_copiers):
        if len(node_copiers) == 1 and isinstance(node_copiers[0], (types.GeneratorType, list, tuple)):
            node_copiers = node_copiers[0]
        self.node_copiers = list(node_copiers)
        for node_copier in self.node_copiers:
            node_copier.multiple = self

    @classmethod
    def copy_node(cls, node, *arg, **kw):
        self = cls(*arg, **kw)
        res = self.copy(node)
        for copier, copy in zip(self.node_copiers, res):
            copy._prism_copier = copier
        return res

    def __getitem__(self, node):
        return self.copy(node)

    def copy(self, node):
        return [copier.copy(node) for copier in self.node_copiers]
        
def mangle_feed_dict(feed_dict, multiple_node_copier):
    res = {}
    batchlen = None
    for key, value in feed_dict.iteritems():
        if batchlen is None:
            batchlen = len(value) % len(multiple_node_copier.node_copiers)
        for idx, node_copier in enumerate(multiple_node_copier.node_copiers):
            if key not in node_copier:
                res[key] = value
                break
            res[node_copier[key]] = value[batchlen * idx:batchlen * (idx+1),:]
    return res

@patch(keras.layers.Layer.add_weight)
@keras.legacy.interfaces.legacy_add_weight_support
def add_weight(origfn,
               self,
               name,
               shape,
               dtype=None,
               initializer=None,
               regularizer=None,
               trainable=True,
               constraint=None):
    """Adds a weight variable to the layer.

    # Arguments
        name: String, the name for the weight variable.
        shape: The shape tuple of the weight.
        dtype: The dtype of the weight.
        initializer: An Initializer instance (callable).
        regularizer: An optional Regularizer instance.
        trainable: A boolean, whether the weight should
            be trained via backprop or not (assuming
            that the layer itself is also trainable).
        constraint: An optional Constraint instance.

    # Returns
        The created weight variable.
    """
    initializer = initializers.get(initializer)
    if dtype is None:
        dtype = K.floatx()
    weight = K.variable(initializer(shape),
                        dtype=dtype,
                        name=name,
                        constraint=constraint)
    if regularizer is not None:
        self.add_loss(regularizer(weight))
    if trainable:
        self._trainable_weights.append(weight)
    else:
        self._non_trainable_weights.append(weight)
    return weight


def slice_batch(x, n_workers, part):
    """
    Divide the input batch into [n_workers] slices, and obtain slice number [part].
    i.e. if len(x)=10, then slice_batch(x, 2, 1) will return x[5:].
    """
    sh = K.shape(x)
    L = sh[0] // n_workers
    if part == n_workers - 1:
        return x[part*L:]
    return x[part*L:(part+1)*L]

def list_devices(sess, filter='/job:worker'):
    return [d.name for d in sess.list_devices()
     if filter in d.name]

def list_tasks(sess, *arg, **kw):
    return list(set('/'.join(part
                             for part in device.split('/')
                             if part == '' or 'job:' in part or 'task:' in part)
                    for device in list_devices(sess, *arg, **kw)))

    
def parallelize(model, ps_server='/job:ps', workers=['/job:worker/task:0']):
    """
    Given a keras [model], return an equivalent model which parallelizes
    the computation over [n_workers] GPUs.

    Each GPU gets a slice of the input batch, applies the model on that slice
    and later the outputs of the models are concatenated to a single tensor, 
    hence the user sees a model that behaves the same as the original.
    """
    with tf.device(ps_server):
        x = Input(model.input_shape[1:], name=model.input_names[0])

    towers = []
    for g, worker in enumerate(workers):
        with tf.device(worker):
            slice_g = Lambda(slice_batch, 
                             lambda shape: shape, 
                             arguments={'n_workers':len(workers), 'part':g})(x)
            towers.append(model(slice_g))

    with tf.device(ps_server):
        merged = Concatenate(axis=0)(towers)

    return Model(inputs=[x], outputs=[merged])
