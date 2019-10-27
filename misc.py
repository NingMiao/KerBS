import tensorflow as tf

def debug_print(content, verbose=False,  tff=True):
  if verbose:
    if tff:
        tf.logging.info(content)
    else:
        print('debug:'+content)
      
      
def advanced_add_to_collections(name, value, name_value):
    name_name=name+'_name'
    if value not in tf.get_collection(name) and name_value not in tf.get_collection(name_name):
        tf.add_to_collections(name, value)
        tf.add_to_collections(name_name, name_value)
        return True
    return False
    
def advanced_get_collection(name, name_value):
    name_name=name+'_name'
    value_dict=dict(zip(tf.get_collection(name_name), tf.get_collection(name)))
    #print(value_dict)
    if name_value in value_dict:
        return value_dict[name_value]
    return None