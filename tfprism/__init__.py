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
    """Format a tensorflow (sub)graph to text as a tree.
    node: the tf op to use as the tree root node, typically your training op or loss op
    depth: optional, maximum numbers of levels to print
    """
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
    """Filter for SingleNodeCopier that merges the output of all gradient
    calculations into a single copy. It also collapses all ops that
    uses (recursively) the gradients into single copies.

    This can be used together with filter_variables to implement data
    parallelism.
    """
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
    """Filter for SingleNodeCopier that keeps variables from being copied.
    Variables are instead shared among all copies made of the graph.
    """
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
    """Returns a filter for SingleNodeCopier that assigns copied nodes to
    a certain device."""
    def set_device(copier, **kw):
        with tf.device(device):
            return copier.apply_next_filter(copier=copier, **kw)
    return set_device

class SingleNodeCopier(object):
    """SingleNodeCopier and MultipleNodeCopier provides the basis
    functionality that TFPrism is built on top of. SingleNodeCopier
    provides a mechanism to copy a (sub)graph, applying a set of
    filters and transforms while doing so.

    MultipleNodeCopier adds the functionality to make multiple copies
    of the graph at the same time, and the filters/transforms being
    aware of this and able to make connections between nodes in the
    graph copies.

    Each node copier keeps track of what nodes it has copied and only
    ever makes a single copy of the whole graph.

        copier = SingleNodeCopier(filters=(filter1,...filterN), prefix="uniq_node_name_prefix_")
        node1_copy = copier.copy(node1)
        node1_copy = copier[node1] # Alternative syntax for the above
        # Will work, even if node2 references node1; node2_copy will
        # reference node1_copy
        node2_copy = copier[node2]
        assert node1 in copier # Check if node1 has been copied
        assert node1_copy is copier[node1]

    It is sometimes usefull to be able to make multiple "copies" of
    the same node withing the destination graph, for example to wrap a
    node copy inside some other operation.

        node_copier.copy(node1, purpose="original_gradient")

    Filters are functions of the form

        def my_filter(node, copier, purpose, **kw):
            if should_node_be_shared(node):
                return node
            elif should_node_be_multiplied(node) and purpose != 'original':
                # Replace node in the copy with something totally else...
                return tf.MatMul(copier.copy(node, purpose='original'), tf.Variable())
            elif should_node_be_on_device(node):
                with tf.device("/job:worker"):
                    return copier.apply_next_filter(node, purpose=purpose, **kw)
            else:
                return None
    """

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
    """SingleNodeCopier and MultipleNodeCopier provides the basis
    functionality that TFPrism is built on top of. MultipleNodeCopier
    provides a mechanism to make multiple copies of a (sub)graph,
    applying a set of filters and transforms across all copies in
    tandem.

    copier = MultipleNodeCopier(
        SingleNodeCopier(filters=filters1, prefix="prefix1_"),
        SingleNodeCopier(filters=filters2, prefix="prefix2_")
        ...)
 
    node_copies = copier.copy(node)

    sess.run(node_copies, feed_dict=copier.mangle_feed_dict(feed_dict))
    """

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
    """Takes a train_step node generated by a optimizer minimize() call
    and parallelizes it using data parallelism on all /job:worker nodes in a cluster.

    Returns (train_steps, copier), where copier.mangle_feed_dict can
    be used as described in the help for MultipleNodeCopier.
    """
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
    """List all device names for a session, filtering by the
    filter string, by default '/job:worker'"""
    return [d.name for d in sess.list_devices()
     if filter in d.name]

def list_tasks(sess, *arg, **kw):
    """List task names for a cluster, filtered by a filter string, by
    default '/job:worker'. This is different from list_devices in that
    it only returns strings on the form '/job:worker/task:5'.
    """
    return list(set('/'.join(part
                             for part in device.split('/')
                             if part == '' or 'job:' in part or 'task:' in part)
                    for device in list_devices(sess, *arg, **kw)))
