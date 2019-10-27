import tensorflow as tf
import numpy as np

def get_new_sense_allocate(allreduce_args):
    #hyper-parameters
    max_component_pre_word=4
    min_component_per_word=1
    reallocate_ratio=0.005
    
    word_num=allreduce_args['word_num']
    eff=(allreduce_args['word_num']+1e-7)/(allreduce_args['efficiency']+1e-7)
    usage=allreduce_args['usage']
    current_allocate=allreduce_args['sense_allocate']
    
    vocab_size=len(word_num)
    min_change_word_num=np.sort(word_num)[vocab_size-int(vocab_size*0.5)]
    
    component_per_word=np.zeros([vocab_size])
    for item in current_allocate:
        component_per_word[item]+=1
    
    #Find useless component:
    component_argsort=np.argsort(usage)
    #Find words that needs new component most
    eff_modified=np.zeros([vocab_size])
    for i in range(vocab_size):
        eff_modified[i]=eff[i]/(word_num[i]**0.5+1e-7)
    word_argsort=np.argsort(eff_modified)
    #Reallocate
    component_argsort_id=0
    change_count=0
    flags=0
    for word_id in word_argsort:
        if component_per_word[word_id]>=max_component_pre_word or word_num[word_id]<=min_change_word_num:
            continue
        while True:
            if change_count>=vocab_size*reallocate_ratio or component_argsort_id>=len(component_argsort):
                flags=1
                break
            component_id=component_argsort[component_argsort_id]
            if component_per_word[current_allocate[component_id]]<=min_component_per_word:
                component_argsort_id+=1
                continue
            component_per_word[current_allocate[component_id]]-=1
            current_allocate[component_id]=word_id
            change_count+=1
            component_argsort_id+=1
            component_per_word[word_id]+=1
            tf.logging.info('Reallocate component: {} to word: {}'.format(component_id, word_id))
            break
        if flags==1:
            break
    tf.logging.info('Reallocate number of this step:{}'.format(change_count))
    return current_allocate
    #return allreduce_args['sense_allocate']

if __name__=='__main__':
    import pickle as pkl
    A=pkl.load(open('dict.pkl','rb'))
    B=get_new_sense_allocate(A)