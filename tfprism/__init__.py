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
    if 'op' in dependencies:
        return hasattr(dependencies['op'], '_merged_gradient')
    for category in ('inputs', 'control_inputs'):
        for item in dependencies[category]:
            if hasattr(item, '_merged_gradient'):
                return True
    return False

def _is_gradient_head(node, copier, **kw):
    if not isinstance(node, tf.Operation) or 'gradients' not in node.name:
        return False
    for output in node.outputs:
        for consumer in output.consumers():
            if 'gradients' in consumer.name:
                return False
    return True
            
def merge_gradients(node, copier, purpose=None, **kw):
    if _is_gradient_head(node=node, copier=copier, purpose=purpose, **kw):
        if purpose == 'original_gradient':
            return None
        else:
            if copier.multiple.node_copiers.index(copier) != 0:
                return copier.multiple.node_copiers[0].copy(node)
            else:
                inputs = [other_copier.copy(node, 'original_gradient')
                          for other_copier in copier.multiple.node_copiers]
                if not node.outputs:
                    with tf.control_dependencies(inputs):
                        res = tf.get_default_graph().create_op(
                            'NoOp',
                            [],
                            [],
                            name=copier.prefix + 'gradient_merge/' + node.name,
                            compute_shapes=True,
                            compute_device=True)
                else:
                    with tf.name_scope(copier.prefix + 'gradient_merge'):
                        res = tf.reduce_mean(
                            tf.stack(
                                axis=0,
                                values=[input.outputs[0] for input in inputs]),
                            0)
                    res._merged_gradient = True
                    res = res.op
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
        res = [copier.copy(node) for copier in self.node_copiers]
        seen = set()
        res = [x for x in res if not (x in seen or seen.add(x))]
        return res

    def mangle_feed_dict(self, feed_dict):
        res = {}
        batchlen = None
        for key, value in feed_dict.iteritems():
            if key not in self.node_copiers[0]:
                res[key] = value
            else:
                if batchlen is None:
                    batchlen = len(value) / len(self.node_copiers)
                # print "Batch for %s: %s of %s examples, %s towers" % (key.name, batchlen, len(value), len(self.node_copiers))
                for idx, node_copier in enumerate(self.node_copiers):
                    assert key in node_copier
                    res[node_copier[key]] = value[batchlen * idx:batchlen * (idx+1),:]
        return res

def distribute_graph_on_all_tasks(node, sess):
    tasks = list_tasks(sess)
    node_copier = MultipleNodeCopier(
        SingleNodeCopier(
            (merge_gradients, filter_variables, set_device(task)),
            task[1:].replace(":", "_") + "/")
        for task in tasks
    )
    nodes = node_copier.copy(node)
    return nodes, node_copier


def list_devices(sess, filter='/job:worker'):
    return [d.name for d in sess.list_devices()
     if filter in d.name]

def list_tasks(sess, *arg, **kw):
    return list(set('/'.join(part
                             for part in device.split('/')
                             if part == '' or 'job:' in part or 'task:' in part)
                    for device in list_devices(sess, *arg, **kw)))
