import numpy as np
import pandas as pd

def make_data(location, n=500, seed=42):
    rng=np.random.default_rng(seed+sum(map(ord,location)))
    hour=rng.integers(0,24,n)
    weekend=rng.integers(0,2,n)
    weather=rng.random(n)
    incident=rng.binomial(1,0.08,n)
    hs=np.sin(2*np.pi*hour/24)
    hc=np.cos(2*np.pi*hour/24)
    base={'Chandigarh-North':520,'Chandigarh-Central':690,'Chandigarh-South':470,'Zirakpur-Corridor':760}.get(location,550)
    rush=100*np.exp(-((hour-9)**2)/7)+125*np.exp(-((hour-18)**2)/9)
    flow=base+85*hs+45*hc-55*weekend-100*weather-155*incident+rush+rng.normal(0,35,n)
    return pd.DataFrame({'hour_sin':hs,'hour_cos':hc,'is_weekend':weekend,'weather_index':weather,'incident_index':incident,'lag_flow':flow+rng.normal(0,45,n),'target_flow':np.clip(flow,20,None)})
