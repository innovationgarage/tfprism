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

@contextlib.contextmanager
def _no_filter(node):
    yield None


@contextlib.contextmanager
def merge_gradients(node):
    if isinstance(node, tf.Operation) and 'gradients_' in node.name:
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=[node, ....])
        yield tf.reduce_mean(grad, 0)
    else:
        yield None
        
@contextlib.contextmanager
def filter_variables(node):
    if isinstance(node, tf.Operation) and node.type == 'VariableV2':
        yield node
    elif (    isinstance(node, tf.Operation)
          and node.type == 'Identity'
          and len(node.inputs) == 1
          and node.inputs[0].op.type == 'VariableV2'):
        yield node
    else:
        yield None

def set_device(device):
    @contextlib.contextmanager
    def set_device(node):
        with tf.device(device):
            yield node
    return set_device

def _chain_2_filters(filter1, filter2):
    @contextlib.contextmanager
    def filter(node):
        with filter1(node) as node1:
            with filter2(node1) as node2:
                yield node2
    return filter

def chain_filters(*filters):
    if len(filters) == 1:
        return filters[0]
    else:
        return _chain_2_filters(filters[0], chain_filters(*filters[1:]))


class NodeCopier(object):
    def __new__(cls, node, filter = _no_filter, prefix = 'copy_'):
        self = object.__new__(cls)
        self.filter = filter
        self.prefix = prefix
        self.mapping = {}
        res = self.copy(node)
        res._copy_mapping = self.mapping
        return res

    def copy(self, node):
        if id(node) in self.mapping:
            return self.mapping[id(node)]
        with self.filter(node) as filtered:
            if filtered is not None:
                res = filtered
            else:
                if isinstance(node, tf.Tensor):
                    res = self.copy_tensor(node)
                elif isinstance(node, tf.Operation):
                    res = self.copy_op(node)
                else:
                    raise ValueError("Unknown node type %s: %s" % (type(node), node))
        self.mapping[id(node)] = res
        return res
                
    def copy_op(self, node):
        attrs = dict(node.node_def.attr)
        if '_class' in attrs:
            attrs['_class'] = tf.AttrValue(list=tf.AttrValue.ListValue(s=[
                name.replace("@", "@%s" % self.prefix) for name in node.get_attr('_class')]))
        with tf.control_dependencies([self.copy(input) for input in node.control_inputs]):
            return tf.get_default_graph().create_op(
                node.type,
                [self.copy(input) for input in node.inputs],
                [to.dtype for to in node.outputs],
                name=self.prefix + node.name,
                attrs=attrs,
                compute_shapes=True,
                compute_device=True
            )
    
    def copy_tensor(self, node):
        idx = node.op.outputs.index(node)
        op = self.copy(node.op)
        return op.outputs[idx]

def mangle_feed_dict(feed_dict, node_copies):
    res = {}
    batchlen = None
    for key, value in feed_dict.iteritems():
        if batchlen is None:
            batchlen = len(value) % len(node_copies)
        for idx, node_copy in enumerate(node_copies):
            if id(key) not in node_copy._copy_mapping:
                res[key] = value
                break
            res[node_copy._copy_mapping[id(key)]] = value[batchlen * idx:batchlen * (idx+1),:]
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
