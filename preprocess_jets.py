import pandas as pd
import numpy as np
import os
import argparse

#completely seperate program
def parse_inputs():
    parser = argparse.ArgumentParser(
        description="JetClass preprocessing + transformer tokenization"
    )

    parser.add_argument("--input_file", "-i", required=True, help="Path to input HDF5 file")
    parser.add_argument(
        "-o", "--output_file", type=str, default=None,
        help="Output processed HDF5 file"
    )

    parser.add_argument(
        "--mode", choices=["single", "triple"],
        default="single",
        help="Tokenization mode"
    )

    parser.add_argument("--n_jets", type = int, default = 300)
    parser.add_argument("--n_const", type=int, default=200)
    parser.add_argument("--lower_q", type = float, default = 0.001)
    parser.add_argument("--upper_q", type = float, default = 1.0)
    parser.add_argument("--n_pt", type=int, default=40, help="Number of pT bins (log-spaced)")
    parser.add_argument("--n_eta", type=int, default=30, help="Number of eta bins")
    parser.add_argument("--n_phi", type=int, default=30, help="Number of phi bins")
    parser.add_argument("--pt_min", type=float, help="Optional pt min for binning (has to be given as log(pt))")
    parser.add_argument("--pt_max", type=float, help="Optional pt max for binning (has to be given as log(pt))")
    parser.add_argument("--eta_min", default = -0.8, type=float, help="Optional eta min for binning (defaults -0.8)")
    parser.add_argument("--eta_max", default = 0.8, type=float, help="Optional eta max for binning (defaults 0.8)")
    parser.add_argument("--phi_min", default = -0.8, type=float, help="Optional phi min for binning (defaults -0.8)")
    parser.add_argument("--phi_max", default = 0.8, type=float, help="Optional phi max for binning (defaults 0.8)")
    parser.add_argument("--combined_tokens", action="store_true", help="Write a single combined token array (legacy). By default separate pt/eta/phi tokens are written.")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--start_jet", default = 0, type = int, help = "At which position in the data set selection should start.")
    parser.add_argument("--tag,", default=None, type = str, help = "Additional tag for labeling the data set")
    parser.add_argument("--all", action = "store_true", help = "If specified the complete data in the specified input_file is used.")

    #set is as default that the tokens are not combined into a single token
    parser.set_defaults(combined_tokens = False)
    parser.set_defaults(all = False)

    args = parser.parse_args()

    return args

#this processes a file with contionus data into binned data with overflow and underflow
def process_h5(input_file: str, output_file: str, options):
    #creates 3 different tables for each feature
    def get_features(data):
        const_pt=data[:,::3].copy()
        d_eta=data[:,1::3].copy()
        d_phi=data[:,2::3].copy()

        return const_pt, d_eta, d_phi

        #return data[:,::3],data[:,1::3],data[:,2::3]

    def truncate_decimals(values, decs= 8):
        return np.trunc(values*10**decs)/(10**decs)

    #computes bin edges accordingly
    #creates n_x - 1 bins for each feature + 1 overflow and +1 underflow bin
    def get_binning(const_pt):

        if (options.pt_max is None) or (options.pt_min is None):
            pt_bins = np.linspace(
                np.quantile(np.log(const_pt[const_pt != 0]), options.lower_q),
                np.quantile(np.log(const_pt[const_pt != 0]), options.upper_q),
                options.n_pt,
            ) 

            print(f"pt_max and pt_min were not set! Choose bins: --pt_min {pt_bins[0]} --pt_max {pt_bins[-1]}")

        else:
            pt_bins = np.linspace(                
                options.pt_min,
                options.pt_max,
                options.n_pt,)

        eta_bins = np.linspace(options.eta_min, options.eta_max, options.n_eta) #(eta_min, ..., eta_max)
        phi_bins = np.linspace(options.phi_min, options.phi_max, options.n_phi) #(phi_min, ..., phi_max)
        
        return pt_bins, eta_bins, phi_bins
    
    #discretize data into the bins (1,..., n_x-1)
    #this reserves 0 as a underflow bin and n_x as a overflow bin --> n_x -1 bins for actual data
    def get_discretized(pt_bins, eta_bins, phi_bins, const_pt, d_eta, d_phi):
        #discetize pt on log scale where const_pt is > 0
        mask = const_pt > 0 #masks only valid pt
        const_pt_disc = np.full_like(const_pt, -1, dtype= np.int16)
        const_pt_disc[mask] = np.digitize(np.log(const_pt[mask]), pt_bins).astype(np.int16)

        #discretize eta and phi linearly
        #automatically assigns 0 if underflow and n_x as overflow
        d_eta_disc = np.digitize(d_eta, eta_bins).astype(np.int16)
        d_phi_disc = np.digitize(d_phi, phi_bins).astype(np.int16)

        # Apply mask every where pt = 0 so we get no invalid const
        const_pt_disc[const_pt == 0] = -1 
        d_eta_disc[const_pt == 0] = -1
        d_phi_disc[const_pt == 0] = -1

        return const_pt_disc, d_eta_disc, d_phi_disc

    #creates a final dataframe from the discretized data
    def get_dataframe(pt, eta, phi):
        #combines the 3 seperate tables of the discretized data back into one big table
        combined = np.stack([pt, eta, phi], -1)
        combined = combined.reshape((-1, options.n_const * 3))
        cols = [
            item
            for sublist in [f"PT_{i},Eta_{i},Phi_{i}".split(",") for i in range(200)]
            for item in sublist
        ]

        df = pd.DataFrame(combined, columns= cols)
        return df

    #this function truncates the decimals and ensures pt ordering
    def check_pt_oredering(pts):

        pts=truncate_decimals(pts)
     
        for i in range(len(pts)):
            if np.all(pts[i, :-1] >= pts[i, 1:])==False:
                #print(pts[i, :])
                print(i)

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    #create output dir if not already existing
    output_dir = os.path.dirname(os.path.abspath(output_file)) or "."
    os.makedirs(output_dir, exist_ok= True)

    print(f"Reads input_file:{input_file}")
    if options.all:
        df = pd.read_hdf(input_file, key = "raw")
    else:
        #reads input file and creates pandas dataframe
        df = pd.read_hdf(input_file, key = "raw", start = options.start_jet, stop = options.start_jet + options.n_jets)


    data = df.to_numpy()

    const_pt, d_eta, d_phi = get_features(data)

    check_pt_oredering(const_pt)

    #create the right bin edges according to the number of desired bins
    pt_bins, eta_bins, phi_bins = get_binning(const_pt)

    #discretize the data
    const_pt_disc, d_eta_disc, d_phi_disc = get_discretized(pt_bins, eta_bins, phi_bins, const_pt, d_eta, d_phi)
    
    disc_df = get_dataframe(const_pt_disc, d_eta_disc, d_phi_disc)
    raw_df = get_dataframe(const_pt, d_eta, d_phi)

    # save the disc data and the continous data in a dataframe
    raw_df.to_hdf(output_file, key="raw", mode="w", complevel=9) #creates a hdf5 file
    disc_df.to_hdf(output_file, key="discretized", mode="r+", complevel=9) #adds to the hdf5 file


def main():
    
    args = parse_inputs()

    # Auto output naming
    if args.output_file is None:
        if args.tag is None: 
            base, ext = os.path.splitext(args.input_file)
            args.output_file = base + "_processed.h5"

            print(f"Output file will be: {args.output_file}")
        if args.tag is not None:
            base, ext = os.path.splitext(args.input_file)
            args.output_file = base + "_" + args.tag + "_processed.h5"

            print(f"Output file will be: {args.output_file}")


    process_h5(args.input_file, args.output_file, options = args)

if __name__ == "__main__":
    main()