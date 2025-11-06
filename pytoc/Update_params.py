import numpy as np

def adam_update(m,v,p,t,beta1,beta2,lr,eps,out_grad):
   
    #print(p)
    #print(out_grad)
    #print(np.shape(out_grad))

    #Note! Squeezing the channel dimention for now for the single channel work. Couldn't find a way around this. DO NOT SQUEEZE IN FUTURE
    out_grad = np.squeeze(out_grad)

    t += 1
    m = beta1*m + (1-beta1) * out_grad
    v = beta2*v + (1-beta2) * (out_grad**2)

    m_hat = m/(1 - beta1 ** t)
    v_hat = v/(1 - beta2 ** t)

    p = p - lr*m_hat/(np.sqrt(v_hat) + eps)

    #print(p)

    return m,v,p,t
    