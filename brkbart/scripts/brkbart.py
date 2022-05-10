from .. import __version__, get_nifti
from ..config import rescan_bart, config
from brkraw import BrukerLoader
import argparse

def main():
    parser = argparse.ArgumentParser(prog='brkbart',
                                     description="BART TOOLBOX integration CLI for BrkRaw (UNC-ZTE)")
    parser.add_argument("-v", "--version", action='version', version='%(prog)s v{}'.format(__version__))

    subparsers = parser.add_subparsers(title='Sub-commands',
                                       description='To run this command, you must specify one of the functions listed'
                                                   'below next to the command. For more information on each function, '
                                                   'use -h next to the function name to call help document.',
                                       help='description',
                                       dest='function',
                                       metavar='command')
    input_str = "input raw Bruker data"
    output_fnm_str = "output filename"
    
    search = subparsers.add_parser("search", help='Search BART password')
    tonii = subparsers.add_parser("tonii", help='Convert a single raw Bruker data into NifTi file(s) using BART')
    
    search.add_argument("path", help="Start location of BART folder search", type=str)
    search.add_argument("depth", help="Search depth", type=str)
    
    tonii.add_argument("-i", "--input", help=input_str, type=str, default=None)
    tonii.add_argument("-o", "--output", help=output_fnm_str, type=str, default=None)
    tonii.add_argument("-s", "--scanid", help="Scan ID, option to specify a particular scan to convert.", type=int)
    tonii.add_argument("-m", "--missing", help="Amount sampling point to skip on trajectory", type=int, default=0)
    tonii.add_argument("-e", "--extend", help="Extend FOV factor", type=float, default=1.0)
    tonii.add_argument("-c", "--core", help="Number of core", type=int, default=1)
    tonii.add_argument("--start", help="start frame for partial reconstruction", type=int, default=None)
    tonii.add_argument("--end", help="end frame for partial reconstruction", type=int, default=None)
    
    args = parser.parse_args()

    if args.function == 'search':
        path = args.path
        depth = int(args.depth)
        rescan_bart(path, depth)
        
    elif args.function == 'tonii':
        path = args.input
        scan_id = args.scanid
        missing = args.missing
        extend = args.extend
        n_thread = args.core
        start = args.start
        end = args.end
        study    = BrukerLoader(path)
        
        if study.is_pvdataset:
            if args.output:
                output = args.output
            else:
                output = '{}_{}'.format(study._pvobj.subj_id,study._pvobj.study_id)
            if scan_id:
                acqpars  = study.get_acqp(int(scan_id))
                scanname = acqpars._parameters['ACQ_scan_name']
                scanname = scanname.replace(' ','-')
                output_fname = '{}-{}-{}'.format(output, scan_id, scanname)
                scan_id = int(scan_id)
        else:
            print('{} is not PvDataset.'.format(path))
        nibobj = get_nifti(path, scan_id, missing, extend, n_thread, start, end)
        nibobj.to_filename(f'{output_fname}.nii.gz')
        print(f'{output_fname}.nii.gz created.')
    else:
        parser.print_help()