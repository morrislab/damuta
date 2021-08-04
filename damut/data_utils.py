def load_sigs(fn, naming_style = 'pcawg', **kwargs):
    styles = ['pcawg', 'cosmic']
    assert naming_style in styles, f'naming_style should be one of {styles}'
    
    
    if naming_style == 'cosmic':
        # read in sigs
        sigs = pd.read_csv(fn, index_col = 0, **kwargs).reindex(mut96)
        # sanity check for matching mut96, should have no NA 
        sel = (~sigs.isnull()).all(axis = 1)
        assert sel.all(),\
            f'invalid signature definitions: null entry for types {list(sigs.index[~sel])}' 
        # convert to pcawg convention
        sigs = sigs.set_index(idx96)
        
    else:
        # read in sigs
        sigs = pd.read_csv(fn, index_col = (0,1), **kwargs).reindex(idx96)
        # sanity check for idx, should have no NA
        sel = (~sigs.isnull()).all(axis = 1)
        assert sel.all(),\
            f'invalid signature definitions: null entry for types {list(sigs.index[~sel])}' 
        
    # check colsums are 1
    sel = np.isclose(sigs.sum(axis=0), 1)
    assert sel.all() ,\
        f'invalid signature definitions: does not sum to 1 in columns {list(sigs.columns[~sel])}' 
        
    return sigs

def split_count(counts, fraction):
    c = (counts * fraction).astype(int)
    frac1 = np.histogram(np.repeat(np.arange(96), c), bins=96, range=(0, 96))[0]
    frac2 = counts - frac1
    assert all(frac2 >= 0) and all(frac1 >= 0)
    return frac1, frac2

def split_by_count(data, fraction=0.8):
    stacked = np.array([split_count(m, fraction) for m in data])
    logging.debug(f'dim counts on split {stacked.shape}')
    return stacked[:,0,:], stacked[:,1,:]

def split_by_S(data, fraction=0.8):
    c = int((data.shape[0] * fraction))
    frac1 = data[0:c]
    frac2 = data[c:(data.shape[0])]
    return frac1, frac2

def get_tau(phi, eta):
    assert len(phi.shape) == 2 and len(eta.shape) == 3
    tau =  np.einsum('jpc,pkm->jkpmc', phi.reshape((-1,2,16)), eta).reshape((-1,96))
    return tau

def get_phis(sigs):
    # for each signature in sigs, get the corresponding phi
    wrapped = sigs.reshape(sigs.shape[0], -1, 16)
    phis = np.hstack([wrapped[:,[0,1,2],:].sum(1), wrapped[:,[3,4,5],:].sum(1)])
    return phis

def get_etas(sigs):
    # for each signature in sigs, get the corresponding eta
    # ordered alphabetically, or pyrimidine last (?)
    wrapped = sigs.reshape(sigs.shape[0], -1, 16)
    etas = wrapped.sum(2)
    return etas#[0,1,2,3,5,4]

def flatten_eta(eta): # eta pkm -> kc
    return np.moveaxis(eta,0,1).reshape(-1, 6)
