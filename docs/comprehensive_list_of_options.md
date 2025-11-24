```
###############################################################################
#                      Monte Carlo eXtreme (MCX) -- CUDA                      #
#          Copyright (c) 2009-2025 Qianqian Fang <q.fang at neu.edu>          #
#                https://mcx.space/  &  https://neurojson.io                  #
#                                                                             #
# Computational Optics & Translational Imaging (COTI) Lab- http://fanglab.org #
#   Department of Bioengineering, Northeastern University, Boston, MA, USA    #
###############################################################################
#    The MCX Project is funded by the NIH/NIGMS under grant R01-GM114365      #
###############################################################################
#  Open-source codes and reusable scientific data are essential for research, #
# MCX proudly developed human-readable JSON-based data formats for easy reuse.#
#                                                                             #
#Please visit our free scientific data sharing portal at https://neurojson.io #
# and consider sharing your public datasets in standardized JSON/JData format #
###############################################################################
$Rev::d20953$v2025.10$Date::2025-10-10 23:08:18 -04$ by $Author::Qianqian Fang$
###############################################################################

usage: mcx <param1> <param2> ...
where possible parameters include (the first value in [*|*] is the default)

== Required option ==
 -f config     (--input)       read an input file in the .json format,if config
                               string starts with '{',it is parsed as an inline
                               JSON input file; if -f is followed by nothing or
                               a single '-', it reads input from stdin via pipe
      or
 -Q/--bench [cube60, skinvessel,...] run a buint-in benchmark specified by name
                               run -Q without parameter to get a list
 -N benchmark  (--net)         get benchmark from NeuroJSON.io, -N only to list
                               benchmark can be dataset URL,or dbname/benchname
                               requires 'curl', install from https://curl.se/

== MC options ==
 -n [0|int]    (--photon)      total photon number (exponential form accepted)
                               max accepted value:9.2234e+18 on 64bit systems
 -r [1|+/-int] (--repeat)      if positive, repeat by r times,total= #photon*r
                               if negative, divide #photon into r subsets
 -b [1|0]      (--reflect)     1 to reflect photons at ext. boundary;0 to exit
 -B '______'   (--bc)          per-face boundary condition (BC), 6 letters for
    /case insensitive/         bounding box faces at -x,-y,-z,+x,+y,+z axes;
                               overwrite -b if given. 
                               each letter can be one of the following:
                               '_': undefined, fallback to -b
                               'r': like -b 1, Fresnel reflection BC
                               'a': like -b 0, total absorption BC
                               'm': mirror or total reflection BC
                               'c': cyclic BC, enter from opposite face

                               if input contains additional 6 letters,
                               the 7th-12th letters can be:
                               '0': do not use this face to detect photon, or
                               '1': use this face for photon detection (-d 1)
                               the order of the faces for letters 7-12 is 
                               the same as the first 6 letters
                               eg: --bc ______010 saves photons exiting at y=0
 -u [1.|float] (--unitinmm)    defines the length unit for the grid edge
 -U [1|0]      (--normalize)   1 to normalize flux to unitary; 0 save raw
 -E [1648335518|int|mch](--seed) set rand-number-generator seed, -1 to generate
                               if an mch file is followed, MCX "replays" 
                               the detected photon; the replay mode can be used
                               to calculate the mua/mus Jacobian matrices
 -z [0|1]      (--srcfrom0)    1 volume origin is [0 0 0]; 0: origin at [1 1 1]
 -k [1|0]      (--voidtime)    when src is outside, 1 enables timer inside void
 -Y [0|int]    (--replaydet)   replay only the detected photons from a given 
                               detector (det ID starts from 1), used with -E 
                               if 0, replay all detectors and sum all Jacobians
                               if -1, replay all detectors and save separately
 -V [0|1]      (--specular)    1 source located in the background,0 inside mesh
 -e [0.|float] (--minenergy)   minimum energy level to trigger Russian roulette
 -g [1|int]    (--gategroup)   number of maximum time gates per run

== GPU options ==
 -L            (--listgpu)     print GPU information only
 -t [16384|int](--thread)      total thread number
 -T [64|int]   (--blocksize)   thread number per block
 -A [1|int]    (--autopilot)   1 let mcx decide thread/block size, 0 use -T/-t
 -G [0|int]    (--gpu)         specify which GPU to use, list GPU by -L; 0 auto
      or
 -G '1101'     (--gpu)         using multiple devices (1 enable, 0 disable)
 -W '50,30,20' (--workload)    workload for active devices; normalized by sum
 -I            (--printgpu)    print GPU information and run program
 --atomic [1|0]                1: use atomic operations to avoid thread racing
                               0: do not use atomic operation (not recommended)

== Input options ==
 -P '{...}'    (--shapes)      a JSON string for additional shapes in the grid.
                               only the root object named 'Shapes' is parsed 
                               and added to the existing domain defined via -f 
                               or --bench
 -j '{...}'    (--json)        a JSON string for modifying all input settings.
                               this input can be used to modify all existing 
                               settings defined by -f or --bench
 -K [1|int|str](--mediabyte)   volume data format, use either a number or a str
       voxel binary data layouts are shown in {...}, where [] for byte,[i:]
       for 4-byte integer, [s:] for 2-byte short, [h:] for 2-byte half float,
       [f:] for 4-byte float; on Little-Endian systems, least-sig. bit on left
                               1 or byte: 0-128 tissue labels
                               2 or short: 0-65535 (max to 4000) tissue labels
                               4 or integer: integer tissue labels 
                              96 or asgn_float: mua/mus/g/n 4xfloat format
                                {[f:mua][f:mus][f:g][f:n]}
                              97 or svmc: split-voxel MC 8-byte format
                                {[n.z][n.y][n.x][p.z][p.y][p.x][upper][lower]}
                              98 or mixlabel: label1+label2+label1_percentage
                                {[label1][label2][s:0-32767 label1 percentage]}
                              99 or labelplus: 32bit composite voxel format
                                {[h:mua/mus/g/n][s:(B15-16:0/1/2/3)(label)]}
                             100 or muamus_float: 2x 32bit floats for mua/mus
                                {[f:mua][f:mus]}; g/n from medium type 1
                             101 or mua_float: 1 float per voxel for mua
                                {[f:mua]}; mus/g/n from medium type 1
                             102 or muamus_half: 2x 16bit float for mua/mus
                                {[h:mua][h:mus]}; g/n from medium type 1
                             103 or asgn_byte: 4-byte gray-levels for mua/s/g/n
                                {[mua][mus][g][n]}; 0-255 mixing prop types 1&2
                             104 or muamus_short: 2-short gray-levels for mua/s
                                {[s:mua][s:mus]}; 0-65535 mixing prop types 1&2
       when formats 99 or 102 is used, the mua/mus values in the input volume
       binary data must be pre-scaled by voxel size (unitinmm) if it is not 1.
       pre-scaling is not needed when using these 2 formats in mcxlab/pmcx
 -a [0|1]      (--array)       1 for C array (row-major); 0 for Matlab array

== Output options ==
 -s sessionid  (--session)     a string to label all output file names
 -O [X|XFEJPMRLSTB](--outputtype) X - output flux, F - fluence, E - energy
    /case insensitive/         J - Jacobian (replay mode),   P - scattering, 
                               event counts at each voxel (replay mode only)
                               M - momentum transfer; R - RF/FD Jacobian
                               L - total pathlength; S - RF/FD mus Jacobian
                               T - time-of-flight*nscat;B - time-of-flight*path
 -d [1|0-3]    (--savedet)     1 to save photon info at detectors; 0 not save
                               2 reserved, 3 terminate simulation when detected
                               photon buffer is filled
 -w [DP|DSPMXVW](--savedetflag)a string controlling detected photon data fields
    /case insensitive/         1 D  output detector ID (1)
                               2 S  output partial scat. even counts (#media)
                               4 P  output partial path-lengths (#media)
                               8 M  output momentum transfer (#media)
                              16 X  output exit position (3)
                              32 V  output exit direction (3)
                              64 W  output initial weight (1)
      combine multiple items by using a string,or add selected numbers together
      by default, mcx only saves detector ID and partial-path data
 -x [0|1]      (--saveexit)    1 to save photon exit positions and directions
                               setting -x to 1 also implies setting '-d' to 1.
                               same as adding 'XV' to -w.
 -X [0|1]      (--saveref)     1 to save diffuse reflectance at the air-voxels
                               right outside of the domain; if non-zero voxels
                               appear at the boundary, pad 0s before using -X
 -m [0|1]      (--momentum)    1 to save photon momentum transfer,0 not to save
                               same as adding 'M' to the -w flag
 -q [0|1]      (--saveseed)    1 to save photon RNG seed for replay; 0 not save
 -M [0|1]      (--dumpmask)    1 to dump detector volume masks; 0 do not save
 -H [1000000] (--maxdetphoton) max number of detected photons
 -S [1|0]      (--save2pt)     1 to save the flux field; 0 do not save
 -F [jnii|...](--outputformat) fluence data output format:
                               mc2 - MCX mc2 format (binary 32bit float)
                               jnii - JNIfTI format (https://neurojson.org)
                               bnii - Binary JNIfTI (https://neurojson.org)
                               nii - NIfTI format
                               hdr - Analyze 7.5 hdr/img format
                               tx3 - GL texture data for rendering (GL_RGBA32F)
    the bnii/jnii formats support compression (-Z) and generate small files
    load jnii (JSON) and bnii (UBJSON) files using below lightweight libs:
      MATLAB/Octave: JNIfTI toolbox   https://neurojson.org/download/jnifty
      MATLAB/Octave: JSONLab toolbox  https://neurojson.org/download/jsonlab
      Python:        PyJData:         https://neurojson.org/download/pyjdata
      JavaScript:    JSData:          https://neurojson.org/download/jsdata
 -Z [zlib|...] (--zip)         set compression method if -F jnii or --dumpjson
                               is used (when saving data to JSON/JNIfTI format)
                               0 zlib: zip format (moderate compression,fast) 
                               1 gzip: gzip format (compatible with *.gz)
                               2 base64: base64 encoding with no compression
                               3 lzip: lzip format (high compression,very slow)
                               4 lzma: lzma format (high compression,very slow)
                               5 lz4: LZ4 format (low compression,extrem. fast)
                               6 lz4hc: LZ4HC format(moderate compression,fast)
 --dumpjson [-,0,1,'file.json'] export all settings,including volume data using
                               JSON/JData (https://neurojson.org) format for
                               easy sharing; can be reused using -f
                               if followed by nothing or '-', mcx will print
                               the JSON to the console; write to a file if file
                               name is specified; by default, prints settings
                               after pre-processing; '--dumpjson 2' prints 
                               raw inputs before pre-processing

== User IO options ==
 -h            (--help)        print this message
 -y [zh_CN,..] (--lang)        select language, followed by nothing to print
 -v            (--version)     print MCX revision number
 -l            (--log)         print messages to a log file instead
 -i            (--interactive) interactive mode

== Debug options ==
 -D [0|int]    (--debug)       print debug information (you can use an integer
  or                           or a string by combining the following flags)
 -D [''|RMPT]                  1 R debug RNG
    /case insensitive/         2 M store photon trajectory info
                               4 P print progress bar
                               8 T save trajectory data only, disable flux/detp
      combine multiple items by using a string,or add selected numbers together

== Additional options ==
 --root         [''|string]    full path to the folder storing the input files
 --gscatter     [1e9|int]      after a photon completes the specified number of
                               scattering events, mcx then ignores anisotropy g
                               and only performs isotropic scattering for speed
 --srcid  [0|-1,0,1,2,..]     -1 simulate multi-source separately;0 all sources
                               together;a positive integer runs a single source
 --internalsrc  [0|1]          set to 1 to skip entry search to speedup launch
 --trajstokes   [0|1]          set to 1 to save Stokes IQUV in trajectory data
 --maxvoidstep  [1000|int]     max distance (in voxel unit) of a photon that
                               can travel before entering the domain, if 
                               launched outside (i.e. a widefield source)
 --maxjumpdebug [10000000|int] when trajectory is requested (i.e. -D M),
                               use this parameter to set the maximum positions
                               stored (default: 1e7)

== Example ==
example: (list built-in benchmarks: -Q/--bench)
       mcx -Q
or (list supported GPUs on the system: -L/--listgpu)
       mcx -L
or (use multiple devices - 1st,2nd and 4th GPUs - together with equal load)
       mcx -Q cube60b -n 1e7 -G 1101 -W 10,10,10
or (use inline domain definition)
       mcx -f input.json -P '{"Shapes":[{"ZLayers":[[1,10,1],[11,30,2],[31,60,3]]}]}'
or (use inline json setting modifier)
       mcx -f input.json -j '{"Optode":{"Source":{"Type":"isotropic"}}}'
or (dump simulation in a single json file)
       mcx -Q cube60planar --dumpjson
or (use -N/--net to browse community-contributed mcx simulations at https://neurojson.io)
       mcx -N
or (run user-shared mcx simulations, see full list at https://neurojson.org/db/mcx)
       mcx -N aircube60
or (print in simplified Chinese using -y/--lang)
       mcx -y zh_CN -Q cube60
or (use -f - to read piped input file modified by shell text processing utilities)
       mcx -Q cube60 --dumpjson | sed -e 's/pencil/cone/g' | mcx -f -
or (download/modify simulations from NeuroJSON.io and run with mcx -f)
       curl -s -X GET https://neurojson.io:7777/mcx/aircube60 | jq '.Forward.Dt = 1e-9' | mcx -f
```