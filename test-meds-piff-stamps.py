def load_meds():
    import ngmix.medsreaders

    return ngmix.medsreaders.NGMixMEDS(
        '/home/esheldon/data/test-medsmom/piff-meds/'
        'DES2214-5914_r6021p01_i_meds-Y6A2_MEDS_V3.fits.fz'
    )


def load_psfcat_and_hsm_cat(iid):
    import numpy as np
    import fitsio
    from glob import glob
    import piff

    si = iid.split('_')[3]

    front = iid[:15]
    expstr = iid[:9]
    r = si[1:1+4]
    p = si[5:5+3]
    pattern = f'/mnt/data/esheldon/DES/meds/Y6A1_v1_desdm-Y6A1v11/DES2214-5914/sources-i/OPS/finalcut/Y6A2_PIFF_V3/r{r}/*/{expstr}/{p}/psf/{front}_*hsmcat*.fits'  # noqa

    # print(pattern)
    flist = glob(pattern)
    assert len(flist) == 1

    catpath = flist[0]
    print('reading:', catpath)
    cat = fitsio.read(catpath)

    pattern = f'/mnt/data/esheldon/DES/meds/Y6A1_v1_desdm-Y6A1v11/DES2214-5914/sources-i/OPS/finalcut/Y6A2_PIFF_V3/r{r}/*/{expstr}/{p}/psf/{front}_*piff-model*.fits'  # noqa

    # print(pattern)
    flist = glob(pattern)
    assert len(flist) == 1

    path = flist[0]
    print('reading:', path)
    pstars = fitsio.read(path, ext='psf_stars')
    pmod = piff.read(path)

    assert np.all(pstars['x'] == cat['x'])
    return cat, pstars, pmod


def run_hsm_gsimage(image):
    """
    Use HSM to measure moments of star image.

    This usually isn't called directly.  The results are accessible as
    star.hsm, which caches the results, so repeated access is efficient.

    :returns: (flux, cenu, cenv, sigma, g1, g2, flag)
    """
    import galsim

    mom = image.FindAdaptiveMom(strict=False)

    sigma = mom.moments_sigma
    shape = mom.observed_shape

    # These are in pixel coordinates.  Need to convert to world coords.
    image_pos = image.true_center
    jac = image.wcs.jacobian(image_pos=image_pos)
    scale, shear, theta, flip = jac.getDecomposition()
    # Fix sigma
    sigma *= scale
    # Fix shear.  First the flip, if any.
    if flip:
        shape = galsim.Shear(g1=-shape.g1, g2=shape.g2)
    # Next the rotation
    shape = galsim.Shear(g=shape.g, beta=shape.beta + theta)
    # Finally the shear
    shape = shear + shape

    flux = mom.moments_amp

    localwcs = image.wcs.local(image_pos)
    center = (
        localwcs.toWorld(mom.moments_centroid) - localwcs.toWorld(image_pos)
    )

    # Do a few sanity checks and flag likely bad fits.
    flag = mom.moments_status
    if flag != 0:
        flag = 1
    if flux < 0:
        flag |= 2
    if center.x**2 + center.y**2 > 1:
        flag |= 4

    T = 2*sigma**2
    return flag, T


def run_hsm(obs):
    """
    Use HSM to measure moments of star image.

    This usually isn't called directly.  The results are accessible as
    star.hsm, which caches the results, so repeated access is efficient.

    :returns: (flux, cenu, cenv, sigma, g1, g2, flag)
    """
    import numpy as np
    import galsim

    image = galsim.ImageD(
        obs.image,
        wcs=obs.jacobian.get_galsim_wcs(),
    )
    weight = galsim.ImageD(obs.weight)

    # image, weight, image_pos = self.data.getImage()
    # Note that FindAdaptiveMom only respects the weight function in a binary
    # sense.  I.e., pixels with non-zero weight will be included in the moment
    # measurement, those with weight=0.0 will be excluded.
    mom = image.FindAdaptiveMom(weight=weight, strict=False)

    sigma = mom.moments_sigma
    shape = mom.observed_shape
    # These are in pixel coordinates.  Need to convert to world coords.
    # jac = image.wcs.jacobian(image_pos=image_pos)
    jac = image.wcs.jacobian()
    scale, shear, theta, flip = jac.getDecomposition()
    # Fix sigma
    sigma *= scale
    # Fix shear.  First the flip, if any.
    if flip:
        shape = galsim.Shear(g1=-shape.g1, g2=shape.g2)
    # Next the rotation
    shape = galsim.Shear(g=shape.g, beta=shape.beta + theta)
    # Finally the shear
    shape = shear + shape

    flux = mom.moments_amp

    cen = (np.array(obs.image.shape) - 1)/2
    image_pos = galsim.PositionD(x=cen[1], y=cen[0])
    localwcs = image.wcs.local(image_pos)
    center = (
        localwcs.toWorld(mom.moments_centroid) - localwcs.toWorld(image_pos)
    )

    # Do a few sanity checks and flag likely bad fits.
    flag = mom.moments_status
    if flag != 0:
        flag = 1
    if flux < 0:
        flag |= 2
    if center.x**2 + center.y**2 > 1:
        flag |= 4

    T = 2*sigma**2
    return flag, T


def get_struct(objid, flags, T, hsmcat, recres):
    import numpy as np
    dt = [
        ('id', 'i8'),
        ('flags', 'i4'),
        ('piff_psf_T', 'f8'),
        ('meds_psf_T', 'f8'),
        ('rec25_cenNone_psf_T', 'f8'),
        ('rec32_cenNone_psf_T', 'f8'),
        ('rec25_cenTrue_psf_T', 'f8'),
        ('rec32_cenTrue_psf_T', 'f8'),
    ]
    st = np.zeros(1, dtype=dt)
    st['id'] = objid
    st['flags'] == flags | recres['flags']
    st['piff_psf_T'] = hsmcat['T_model']
    st['meds_psf_T'] = T

    for key in recres:
        if key == 'flags':
            continue
        st[key] = recres[key]

    return st


def doplot(m, pstars, pngname):
    import proplot as pplt

    fig, ax = pplt.subplots()

    ax.scatter(m['ra'], m['dec'], ms=1)

    ax.scatter(
        pstars['ra'] * 15,
        pstars['dec'],
        m='o',
        facecolors='none',
        edgecolors='firebrick',
    )
    print('writing:', pngname)
    fig.savefig(pngname)


def main():
    import numpy as np
    import fitsio
    import esutil as eu
    from esutil.numpy_util import combine_arrlist

    m = load_meds()
    ii = m.get_image_info()
    pi = m._fits['psf_info'].read()

    w = np.where(m['file_id'] > 0)
    ufile_ids = np.unique(m['file_id'][w])

    ntest = 200
    file_ids_to_test = ufile_ids[:ntest]
    nkeep = file_ids_to_test.size

    for file_id_index, file_id in enumerate(file_ids_to_test):
        print('-' * 70)
        print(f'{file_id_index+1} / {nkeep}')

        iid = ii['image_id'][file_id]
        expnum = int(iid[1:9])
        ccdnum = int(iid[13:15])

        wpi, = np.where(
            (pi['expnum'] == expnum)
            & (pi['ccdnum'] == ccdnum)
        )
        assert wpi.size == 1

        fpart = pi['filename'][wpi[0]][:24]
        fname = f'compare-DES2214-5914-{fpart}.fits'
        pngname = fname.replace('.fits', '-radec.png')

        w = np.where(m['file_id'] == file_id)
        print(f'found {w[0].size} objects in {iid}')

        hsmcat, pstars, pmod = load_psfcat_and_hsm_cat(pi['filename'][wpi[0]])

        htm = eu.htm.HTM(depth=12)
        mpstars, mmeds, dist = htm.match(
            pstars['ra'] * 15, pstars['dec'],
            m['ra'][w[0]], m['dec'][w[0]],
            radius=1.0/3600,
        )
        print(mpstars.size, pstars.size)
        doplot(m, pstars, pngname)
        if mpstars.size == 0:
            continue

        indices = w[0][mmeds]
        cutids = w[1][mmeds]

        num = mmeds.size
        dlist = []
        for i in range(num):
            wp = mpstars[i]
            oindex = indices[i]
            cutid = cutids[i]

            objid = m['id'][oindex]

            obs = m.get_obs(oindex, cutid)
            flags, T = run_hsm(obs.psf)

            row = m['orig_row'][oindex, cutid]
            col = m['orig_col'][oindex, cutid]

            results = {'flags': 0}
            for stamp_size in (25, 32):
                for center in (True, None):
                    recim = pmod.draw(
                        x=col, y=row, GI_COLOR=m['psf_color'][oindex],
                        chipnum=int(pstars['chipnum'][wp]),
                        stamp_size=stamp_size,
                        center=center,
                    )
                    tflags, tT = run_hsm_gsimage(recim)
                    results['flags'] |= tflags
                    results[f'rec{stamp_size}_cen{center}_psf_T'] = tT

            st = get_struct(
                objid=objid,
                flags=flags,
                T=T,
                hsmcat=hsmcat[wp],
                recres=results,
            )
            dlist.append(st)

        st = combine_arrlist(dlist)
        print('writing', st.size, 'to', fname)
        fitsio.write(fname, st)


main()
