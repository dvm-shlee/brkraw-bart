"""_summary_

Raises:
    Exception: _description_

Returns:
    _type_: _description_
"""
import sys
import os
import time
import tempfile as tmp
import multiprocessing as mp
import numpy as np
import nibabel as nib
from scipy.interpolate import interp1d
from tqdm import tqdm
import brkraw as brk
import paralexe as pe
from . import config
from . import utils
from nibabel.affines import from_matvec, to_matvec
from brkraw.lib.orient import reversed_pose_correction, build_affine_from_orient_info, apply_rotate

__version__ = "0.0.3"

try:
    __toolbox_path = config.config.get('Default', 'toolbox_path')
    sys.path.insert(0, os.path.join(__toolbox_path, 'python'))
    os.environ['TOOLBOX_PATH'] = __toolbox_path
    import cfl
except:
    print("BART not installed")


def parse_acqp(rawobj, scan_id):
    # acqp parameter parsing
    acqp = rawobj.get_acqp(scan_id).parameters
    nr = acqp['NR']
    ni = acqp['NI']
    nae = acqp['NAE']  # number of average
    ns = acqp['NSLICES']
    acq_jobs = acqp['ACQ_jobs'][0]
    sz = [acq_jobs[0]/2, acq_jobs[3]/(nr*nae), acq_jobs[-2]]

    wordtype = brk.lib.reference.WORDTYPE[f'_{"".join(acqp["ACQ_word_size"].split("_"))}_SGN_INT']
    byteorder = brk.lib.reference.BYTEORDER[f'{acqp["BYTORDA"]}Endian']

    dtype_code = np.dtype(f'{byteorder}{wordtype}')
    fid_shape = np.array([2] + sz + [ns]).astype(int).tolist()
    num_frames = int(nr*ni/ns)
    buffer_size = np.prod(fid_shape)*dtype_code.itemsize

    return dict(fid_shape=fid_shape,
                dtype_code=dtype_code,
                num_frames=num_frames,
                buffer_size=buffer_size)


def get_traj(rawobj, scan_id):
    method = rawobj.get_method(scan_id).parameters
    traj_shape = np.array(
        [3, method['PVM_Matrix'][0]/2, method['NPro']]).astype(int)
    traj = np.frombuffer(rawobj._pvobj.get_traj(scan_id),
                         np.double).reshape(traj_shape, order='F')
    return traj


def upsample_traj(traj, spoke_add):
    gap = spoke_add+1
    traj_corrected = np.zeros(
        (traj.shape[0], traj.shape[1], traj.shape[2]+((traj.shape[2]-1)*spoke_add)))
    x = np.arange(traj_corrected.shape[-1])
    idx = x[::gap]

    for grad in range(traj.shape[0]):
        for samp in range(traj.shape[1]):
            traj_corrected[grad, samp, ::gap] = traj[grad, samp, :]
            interp = interp1d(x[idx], traj_corrected[grad,
                              samp, idx], bounds_error=False)
            traj_corrected[grad, samp, :] = interp(x)
    return traj_corrected


def ramp_time_correction(traj, ramp_time, grad_res):

    # Initializing modified trajectory
    traj_ramp = np.zeros((traj.shape))
    k_space = np.zeros(int(np.ceil(ramp_time/grad_res)))

    step_time_ramp = np.linspace(0, ramp_time, len(k_space))
    step_time_acq = np.linspace(0, ramp_time, traj.shape[1])

    size_traj_x = traj.shape[0]
    size_traj_z = traj.shape[-1]
    for i in tqdm(range(size_traj_z)):

        end_idx = 0 if i == size_traj_z - 1 else i + 1

        for j in range(size_traj_x):
            # Gradient amplitude of last projection
            start = traj[j, -1, i]

            # Gradient amplitude of current projection
            end = traj[j, -1, end_idx]

            # gradient amplitude steps from previous amplitude to next gradient amplitude
            grad = np.linspace(start/ramp_time, end/ramp_time, len(k_space))

            # Calaculating k-space coordinate at very gradient incrementation interval
            for k in range(len(k_space)):
                if k == 0:
                    k_space[k] = grad[k] * grad_res
                else:
                    k_space[k] = grad[k] * grad_res + k_space[k-1]

            # Interpolating acquired data coordinates onto k-space
            k_traj = np.interp(step_time_acq, step_time_ramp, k_space)

            traj_ramp[j, :, end_idx] = k_traj
    return traj_ramp


def calc_trajectory(rawobj, scan_id, params, ext_factor, ramp_correction):
    method = rawobj.get_method(scan_id).parameters
    over_samp = method['OverSampling']
    mat_size = np.array(method['PVM_Matrix'])

    num_pts = int((mat_size[0] * over_samp) / 2)
    smp_idx = np.arange(0, int(mat_size[0] / 2))
    smp_idx_oversmp = np.array(
        [c+s for s in smp_idx for c in np.linspace(0, 1-1/over_samp, over_samp)])

    # trajectory
    traj = get_traj(rawobj, scan_id)

    # calculate trajectory oversampling
    # axis2 = number of projection
    traj_oversmp = np.zeros((3, num_pts, params['fid_shape'][2]))
    for i, coord in enumerate(traj):
        # Mean across rows of differnce across rows
        step = np.mean(np.diff(coord, 1, axis=0), axis=0)
        coord = np.insert(coord, -1, coord[-1, :], axis=0)
        coord[-1, :] = coord[-2, :] + step
        func = interp1d(np.append(smp_idx, smp_idx[-1]+1), coord, axis=0)

        # Evaluating trajectory at finer intervals
        traj_oversmp[i, :, :] = func(smp_idx_oversmp)

    traj_oversmp[:, :, ::2] = -traj_oversmp[:, :, 1::2]  # correct direction

    traj_oversmp = np.multiply(traj_oversmp, mat_size[:, np.newaxis, np.newaxis]
                               .repeat(traj_oversmp.shape[1], 1)
                               .repeat(traj_oversmp.shape[2], 2))

    proj_order = np.concatenate([np.arange(0, params['fid_shape'][2], 2),
                                 np.arange(1, params['fid_shape'][2], 2)])

    # apply extend FOV factor
    traj_adjusted = traj_oversmp[:, :, proj_order] * ext_factor

    if ramp_correction:
        # ramp time
        ramp_time = method['RampTime']
        # gradient amplitude resolusion in milisecond
        grad_res = rawobj.get_method(
            scan_id=scan_id).parameters['GradRes'] / 1000
        traj_adjusted = ramp_time_correction(
            traj_adjusted, ramp_time, grad_res)
    return traj_adjusted


def recon_dataobj(rawobj, scan_id, missing, ext_factor, n_thread, crop_range, ramp_correction):
    # prep basename of temporary file
    temp_basename = tmp.NamedTemporaryFile().name

    # acqp parameter parsing
    print('Converting trajectory to cfl...', end='')
    params = parse_acqp(rawobj, scan_id)
    traj = calc_trajectory(rawobj, scan_id, params, ext_factor, ramp_correction)
    traj_path = f'{temp_basename}_traj'
    if missing > 0:
        traj = traj[:, missing:, ...]

    cfl.writecfl(traj_path, traj)
    del traj  # clear memory
    print('done')

    # data size
    data_size = np.array(rawobj.get_matrix_size(scan_id, 1))
    data_size[:3] = np.ceil(data_size[:3] * ext_factor)

    volm_fnames = []
    oput_fnames = []
    pvobj = rawobj._pvobj

    print('Converting fid to cfl...', end='')
    with pvobj._open_object(pvobj._fid[scan_id]) as f:
        if params['num_frames'] > 1:
            pnt_frames = len(str(params['num_frames']))
            start = crop_range[0] if crop_range[0] is not None else 0
            end = crop_range[1] if crop_range[1] is not None else params['num_frames'] - 1

            for frame in range(params['num_frames']):
                if frame >= start and frame <= end:
                    buffer = f.read(params['buffer_size'])
                    v = np.frombuffer(buffer, params['dtype_code']).reshape(
                        params['fid_shape'], order='F')
                    v = (v[0]+1j*v[1])[np.newaxis, ...]
                    if missing > 0:
                        v = v[:, missing:, ...]
                    volm_path = f'{temp_basename}_volm{str(frame).zfill(pnt_frames)}'
                    oput_path = f'{temp_basename}_oput{str(frame).zfill(pnt_frames)}'
                    cfl.writecfl(volm_path, v)
                    volm_fnames.append(volm_path)
                    oput_fnames.append(oput_path)
                    del v, buffer  # clear memory
        else:
            buffer = f.read()
            v = np.frombuffer(buffer, params['dtype_code']).reshape(
                params['fid_shape'], order='F')
            v = (v[0]+1j*v[1])[np.newaxis, ...]
            volm_path = f'{temp_basename}_volm'
            oput_path = f'{temp_basename}_oput'
            cfl.writecfl(volm_path, v)
            volm_fnames.append(volm_path)
            oput_fnames.append(oput_path)
            del v, buffer  # clear memory
    print('done')

    print(f'Reconstruction...[num_threads: {n_thread}]')
    dataobj = []
    mng = pe.Manager()
    cmd = f'{__toolbox_path}/bart nufft -i *[traj_path] *[volm_fnames] *[oput_fnames]'
    mng.set_cmd(cmd)
    mng.set_arg(label='traj_path', args=[traj_path] * len(volm_fnames))
    mng.set_arg(label='volm_fnames', args=volm_fnames)
    mng.set_arg(label='oput_fnames', args=oput_fnames)
    mng.schedule(n_thread=int(n_thread))
    mng.submit('background')

    # wait until finished

    mng.schd.check_progress()
    while mng.schd.is_alive():
        if len(mng.schd._failed_workers):
            if len(mng.schd._failed_workers[0]):
                for wid in mng.schd._failed_workers[0]:
                    print(mng._workers[wid].output, file=sys.stderr)
                for p in volm_fnames:
                    for rpath in [f'{p}.cfl', f'{p}.hdr']:
                        os.remove(rpath)
                os.remove(f'{traj_path}.cfl')
                os.remove(f'{traj_path}.hdr')
                raise Exception('Error on subprocessor...')
        time.sleep(1)
    print(mng.audit())
    print('Data assembling...', end='')
    os.remove(f'{traj_path}.cfl')
    os.remove(f'{traj_path}.hdr')

    if len(oput_fnames) > 1:
        for i, p in enumerate(oput_fnames):
            dataobj.append(np.abs(cfl.readcfl(p))[..., np.newaxis])
            for rpath in [f'{p}.cfl', f'{p}.hdr', f'{volm_fnames[i]}.cfl', f'{volm_fnames[i]}.hdr']:
                os.remove(rpath)
        dataobj = np.concatenate(dataobj, axis=-1)
    else:
        dataobj = np.abs(cfl.readcfl(oput_fnames[0]))
        for rpath in [f'{oput_fnames[0]}.cfl', f'{oput_fnames[0]}.hdr', f'{volm_fnames[0]}.cfl', f'{volm_fnames[0]}.hdr']:
            os.remove(rpath)
    print('done')

    return dataobj


def scale_pose(rmat, pose, ext_factor=1):
    if ext_factor != 1:
        # scale_matrix
        smat = np.eye(3)*ext_factor
        # scale_and_rotation_matrix
        srmat = smat.dot(rmat)
        # correct pose
        pose = srmat.T.dot(pose)
        pose = rmat.dot(pose)
    return pose


def build_affine_from_orient_info(resol, rmat, pose,
                                  subj_pose, subj_type, slice_orient):
    if slice_orient in ['axial', 'sagital']:
        resol = np.diag(np.array(resol))
    else:
        resol = np.diag(np.array(resol) * np.array([1, 1, -1]))
    rmat = rmat.T.dot(resol)
    
    affine = from_matvec(rmat, pose)

    # convert space from image to subject
    # below positions are all reflect human-based position
    if subj_pose:
        if subj_pose == 'Head_Supine':
            affine = apply_rotate(affine, rad_z=np.pi)
        elif subj_pose == 'Head_Prone':
            pass
        # From here, not urgent, extra work to determine correction matrix needed.
        elif subj_pose == 'Head_Left':
            affine = apply_rotate(affine, rad_z=np.pi/2)
        elif subj_pose == 'Head_Right':
            affine = apply_rotate(affine, rad_z=-np.pi/2)
        elif subj_pose in ['Foot_Supine', 'Tail_Supine']:
            affine = apply_rotate(affine, rad_x=np.pi)
        elif subj_pose in ['Foot_Prone', 'Tail_Prone']:
            affine = apply_rotate(affine, rad_y=np.pi)
        elif subj_pose in ['Foot_Left', 'Tail_Left']:
            affine = apply_rotate(affine, rad_z=np.pi/2)
        elif subj_pose in ['Foot_Right', 'Tail_Right']:
            affine = apply_rotate(affine, rad_z=-np.pi/2)
        else:  # in case Bruker put additional value for this header
            raise Exception('NotIntegrated')

    if subj_type != 'Biped':
        # correct subject space if not biped (human or non-human primates)
        # not sure this rotation is independent with subject pose, so put here instead last
        affine = apply_rotate(affine, rad_x=-np.pi/2, rad_y=np.pi)
    return affine


def calc_affine(rawobj, scan_id, reco_id, ext_factor=1, preclinical_view=True):
    visu_pars = rawobj.get_visu_pars(scan_id, reco_id)
    method = rawobj.get_method(scan_id)

    is_reversed = True if rawobj._get_disk_slice_order(visu_pars) == 'reverse' else False
    slice_info = rawobj._get_slice_info(visu_pars)
    spatial_info = rawobj._get_spatial_info(visu_pars)
    orient_info = rawobj._get_orient_info(visu_pars, method)
    slice_orient_map = {0: 'sagital', 1: 'coronal', 2: 'axial'}
    subj_pose = orient_info['subject_position']
    if preclinical_view:
        subj_type = None
    else:
        subj_type = orient_info['subject_type']

    sidx = orient_info['orient_order'].index(2)
    slice_orient = slice_orient_map[sidx]
    resol = spatial_info['spatial_resol'][0]
    rmat = orient_info['orient_matrix']
    pose = orient_info['volume_position']
    if is_reversed:
        distance = slice_info['slice_distances_each_pack']
        pose = reversed_pose_correction(pose, rmat, distance)
    pose = scale_pose(rmat, pose, ext_factor)
    affine = build_affine_from_orient_info(resol, rmat, pose, subj_pose, subj_type, slice_orient)
    return affine


def get_nifti(path, scan_id, missing=0, ext_factor=1, n_thread=1, start=None, end=None, ramp_correction=True, preclinical_view=True):
    rawobj = brk.load(path)
    dataobj_raw = recon_dataobj(
        rawobj, scan_id, missing, ext_factor, n_thread, [start, end], ramp_correction)
    # Datatype
    dataobj = ((dataobj_raw / dataobj_raw.max()) * 2**16).astype(np.uint16)
    del dataobj_raw  # clear memory

    # Gradient direction correction
    grad_matrix = rawobj.get_acqp(
        scan_id).parameters['ACQ_GradientMatrix'][0].T
    grad_matrix = np.round(grad_matrix, decimals=0)
    axis_order = np.arange(dataobj.ndim)
    axis_order[:3] = tuple([int(np.squeeze(np.nonzero(ax)))
                           for ax in grad_matrix])
    flip_axis = np.nonzero(grad_matrix.sum(0) < 0)[0].tolist()
    dataobj = dataobj.transpose(axis_order)
    dataobj = np.flip(dataobj, flip_axis)

    # Position correction
    axis_order[:3] = [0, 2, 1]
    dataobj = dataobj.transpose(axis_order)
    dataobj = np.flip(dataobj, 1)

    # NIFTI conversion
    affine = calc_affine(rawobj, scan_id, 1, ext_factor, preclinical_view)

    visu_pars = rawobj.get_visu_pars(scan_id, 1)
    temporal_resol = rawobj._get_temp_info(visu_pars)['temporal_resol']
    temporal_resol = float(temporal_resol) / 1000

    nibobj = nib.Nifti1Image(dataobj, affine)
    if dataobj.ndim > 3:
        nibobj.header.set_xyzt_units(xyz=2, t=8)
        nibobj.header['pixdim'][4] = temporal_resol
    nibobj.set_sform(affine, 0)
    nibobj.set_qform(affine, 1)
    return nibobj
