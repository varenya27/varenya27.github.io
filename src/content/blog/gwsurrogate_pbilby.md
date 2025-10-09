---
title: 'GW150914 Parameter Estimation with Surrogate Models'
description: 'Lorem ipsum dolor sit amet'
pubDate: 'Oct 03 2025'
heroImage: '../../assets/NRHybSur3dq8_waveform_time_domain.png'
---



|                    |                 |
| ------------------ | --------------- |
| PE Code            | parallel_bilby  |
| Waveform Generator | `gwsurrogate`/  |
| Implementation     | `gwsignal`-like |
| Waveform Model     | NRHybSur3dq8    |
| Cluster            | Unity           |

# gwsurr.py
<details>
<summary>Show Code</summary>

```python
try:
    import gwsurrogate as gwsurr
except ImportError:
    print("The gwsurrogate package has failed to load, exiting")

from importlib_metadata import metadata
import lal
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from gwpy.timeseries import TimeSeries
import astropy.units as u

from lalsimulation.gwsignal.core.waveform import CompactBinaryCoalescenceGenerator
import lalsimulation.gwsignal.core.gw as gw
from lalsimulation.gwsignal.core.utils import add_params_units
import lalsimulation as lalsim 

# ignore spin magnitude outside training space warnings
import warnings
warnings.filterwarnings('ignore', message='.*Spin')

class NRHybSur3dq8_gwsurr(CompactBinaryCoalescenceGenerator):
    def __init__(self, **kwargs):
        self.sur = gwsurr.LoadSurrogate("NRHybSur3dq8")
        self._update_domains()

    @property
    def metadata(self):
        metadata = {
            "type": "aligned-spin",
            "f_ref_spin": True,
            "modes": True,
            "polarizations": True,
            "implemented_domain": "time",
            "approximant" : 'NRSurr',
            "implementation" : "Python",
            "conditioning_routines" : 'gwsignal'
        }
        return metadata

    def generate_td_modes(self, **parameters):
        self.parameter_check(units_sys='Cosmo', **parameters)
        self.waveform_dict = self._strip_units(self.waveform_dict)
        fmin, dt = self.waveform_dict["f22_start"], self.waveform_dict["deltaT"]
        f_ref = self.waveform_dict["f22_ref"]

        m1, m2 = self.waveform_dict["mass1"], self.waveform_dict["mass2"]
        s1z= self.waveform_dict["spin1z"]
        s2z= self.waveform_dict["spin2z"]
        chi1 = np.array( [
            0.,
            0.,
            s1z,
        ])
        chi2 = np.array( [
            0.,
            0.,
            s2z,
        ])
        dist = self.waveform_dict["distance"]
        q = m1 / m2  # This is the gwsurrogate convention, q=m1/m2>=1
        if q < 1.0:
            q = 1 / q


        # VU: reduce fmin to make sure tapering doesn't remove signal: [cf L#1046 in SimInspiral.c]
        extra_cycles = 3. 
        extra_time_fraction = 0.1
        m1_kg = m1 * lal.MSUN_SI
        m2_kg = m2 * lal.MSUN_SI
        tchirp = lalsim.SimInspiralChirpTimeBound(fmin, m1_kg, m2_kg, s1z, s2z)
        s = lalsim.SimInspiralFinalBlackHoleSpinBound(s1z,s2z)
        tmerge = lalsim.SimInspiralMergeTimeBound(m1_kg,m2_kg)+lalsim.SimInspiralRingdownTimeBound(m1_kg+m2_kg,s)
        textra = extra_cycles / fmin
        fstart = lalsim.SimInspiralChirpStartFrequencyBound((1.+extra_time_fraction)*tchirp+tmerge+textra,m1_kg,m2_kg)

        times, h, dyn = self.sur(
            q,
            chi1,
            chi2,
            dt=dt,
            f_low=fstart,
            f_ref=f_ref,
            units="mks",  # Output in SI units
            M=m1 + m2,  # In solar masses
            dist_mpc=dist/1e6,  # In Mpc
        )

        hlm = self._to_gwpy_series(h, times)
        return gw.GravitationalWaveModes(hlm)

    def generate_td_waveform(self, **parameters):
        # VU: added pi/2-phi_ref to match LALSuite convention
        theta, phi = parameters['inclination'], (np.pi/2-parameters['phi_ref'].value)*u.rad
        hlm = self.generate_td_modes(**parameters)
        hp, hc = hlm(theta, phi)
        hp, hc = TimeSeries(hp, name='hplus'), TimeSeries(hc, name='hcross')
        return hp, hc

    def generate_fd_polarizations_from_td(self, **parameters):
        # Adjust deltaT depending on sampling rate
        fmax = parameters["f_max"].value
        f_nyquist = fmax
        deltaF = 0
        if "deltaF" in parameters.keys():
            deltaF = parameters["deltaF"].value

        if deltaF != 0:
            n = int(np.round(fmax / deltaF))
            if n & (n - 1):
                chirplen_exp = np.frexp(n)
                f_nyquist = np.ldexp(1, int(chirplen_exp[1])) * deltaF

        deltaT = 0.5 / f_nyquist
        parameters["deltaT"] = deltaT*u.s


        hp_,hc_ = self.generate_td_waveform(**parameters)
        # VU: set epoch to merger time according to surrogate convention (instead of start time)
        epoch = lal.LIGOTimeGPS(
            hp_.times[np.abs(np.array(hp_.times)).argmin()].value
        )
        hp = lal.CreateREAL8TimeSeries(
            "hplus", epoch, 0, parameters["deltaT"].value, lal.DimensionlessUnit, len(hp_)
        )
        hc = lal.CreateREAL8TimeSeries(
            "hcross", epoch, 0, parameters["deltaT"].value, lal.DimensionlessUnit, len(hc_),
        )

        hp.data.data = hp_.value
        hc.data.data = hc_.value

        m1 = parameters['mass1'].value
        m2 = parameters['mass2'].value
        s1z= parameters['spin1z'].value
        s2z= parameters['spin2z'].value 
        fmin = parameters['f22_start'].value
        extra_cycles = 3. 
        extra_time_fraction = 0.1
        m1_kg = m1 * lal.MSUN_SI
        m2_kg = m2 * lal.MSUN_SI
        tchirp = lalsim.SimInspiralChirpTimeBound(fmin, m1_kg, m2_kg, s1z, s2z)
        s = lalsim.SimInspiralFinalBlackHoleSpinBound(s1z,s2z)
        tmerge = lalsim.SimInspiralMergeTimeBound(m1_kg,m2_kg)+lalsim.SimInspiralRingdownTimeBound(m1_kg+m2_kg,s)
        textra = extra_cycles / fmin
        fstart = lalsim.SimInspiralChirpStartFrequencyBound((1.+extra_time_fraction)*tchirp+tmerge+textra,m1_kg,m2_kg)

        lalsim.SimInspiralTDConditionStage1(hp,hc, extra_time_fraction * tchirp +textra,fmin)

        fisco = 1.0 / ( (6.0**1.5) * lal.PI * (m1_kg + m2_kg) * lal.MTSUN_SI / lal.MSUN_SI);

        lalsim.SimInspiralTDConditionStage2(hp,hc, fmin,fisco)

        if deltaF == 0:
            chirplen = hp.data.length
            chirplen_exp = np.frexp(chirplen)
            chirplen = int(np.ldexp(1, chirplen_exp[1]))
            deltaF = 1.0 / (chirplen * deltaT)
            parameters["deltaF"] = deltaF

        else:
            chirplen = int(1.0 / (deltaF * deltaT))

        lal.ResizeREAL8TimeSeries(hp, hp.data.length - chirplen, chirplen)
        lal.ResizeREAL8TimeSeries(hc, hc.data.length - chirplen, chirplen)

        # FFT - Using LAL routines
        hptilde = lal.CreateCOMPLEX16FrequencySeries(
            "FD H_PLUS",
            hp.epoch,
            0.0,
            deltaF,
            lal.DimensionlessUnit,
            int(chirplen / 2.0 + 1),
        )
        hctilde = lal.CreateCOMPLEX16FrequencySeries(
            "FD H_CROSS",
            hc.epoch,
            0.0,
            deltaF,
            lal.DimensionlessUnit,
            int(chirplen / 2.0 + 1),
        )

        plan = lal.CreateForwardREAL8FFTPlan(chirplen, 0)
        lal.REAL8TimeFreqFFT(hctilde, hc, plan)
        lal.REAL8TimeFreqFFT(hptilde, hp, plan)

        # print('DBUG', type(hptilde),hptilde)
        return hptilde.data, hctilde.data
       
    def _to_gwpy_series(self, modes_dict, times):
        """
        Iterate over the dict and return a dict of gwpy TimeSeries objects
        """
        gwpy_dict = {}
        for ellm, mode in modes_dict.items():
            gwpy_dict[ellm] = TimeSeries(mode, times=times, name='h_%i_%i'%(ellm[0], ellm[1]))
        return gwpy_dict


    def _strip_units(self, waveform_dict):
        new_dc = {}
        for key in waveform_dict.keys():
            if isinstance(waveform_dict[key], u.Quantity):
                new_dc[key] = waveform_dict[key].value
            else:
                new_dc[key] = waveform_dict[key]
        return new_dc
```

</details>

- gwsignal-like implementation that makes calls to gwsurrogate for the waveform at some parameters given by the PE code
- The class `NRHybSur3dq8_gwsurr` inherits from `lalsimulation.gwsignal.core.waveform.CompactBinaryCoalescenceGenerator` and returns a waveform that is ready to go to bilby's likelihood function
- Add file location to python path, eg. add to `.bashrc`:
```bash
export PYTHONPATH="$HOME/src/new-waveforms-interface/python_interface/gwsignal/models:$PYTHONPATH"
```
- Key methods in the class:
    1. `generate_td_modes` makes calls to gwsurrogate; check for parameter consistencies here!
    2. `generate_td_waveform` $h_+-ih_\times=\sum h_{lm}(t;\lambda)_{-2}Y_{lm}(\theta,\phi)$
    3. `generate_fd_polarizations_from_td` 
        1. adjusts `deltaT`
        2. sets `epoch` as merger time
        3. computes a frequency lower than `f_low`; taper between these two
        4. resizes and returns FFTs of td waveforms

# gwsurr_wrappers.py 
<details>
<summary>Show Code</summary>

```python 
from gwsurr import NRHybSur3dq8_gwsurr 
from astropy import units as u
import numpy as np 

gen = NRHybSur3dq8_gwsurr()

from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
from bilby.gw.waveform_generator import WaveformGenerator

def parameter_conversion(parameters):
    mass_1,mass_2 = chirp_mass_and_mass_ratio_to_component_masses(parameters['chirp_mass'],parameters['mass_ratio'])
    params = {
        'mass1':mass_1,
        'mass2':mass_2,
        'spin1z':parameters['chi_1'],
        'spin2z':parameters['chi_2'],
        'distance':parameters['luminosity_distance'],
        'inclination':parameters['theta_jn'],
        'phi_ref':parameters['phase'],
    }
    keys=[]
    for key in params.keys():
        if key not in parameters.keys():
            keys.append(key)
    return params, keys

def get_waveform_generator(**kwargs):
    # bilby sometimes defaults to the inbuilt BBH parameter conversion function, which we don't want
    if not kwargs['parameter_conversion'] is parameter_conversion:
        print(f"PROG Updating parameter conversion function from {kwargs['parameter_conversion']} to {parameter_conversion}")
        kwargs['parameter_conversion']=parameter_conversion

    return WaveformGenerator(**kwargs)

def NRHybSur3dq8_wrapper(freqs, mass1,mass2,spin1z,spin2z,distance,inclination,phi_ref,**waveform_arguments):
    hp,hc =  gen.generate_fd_polarizations_from_td(
        mass1=mass1*u.Msun,
        mass2=mass2*u.Msun,
        spin1z=spin1z*u.dimensionless_unscaled,
        spin2z=spin2z*u.dimensionless_unscaled,
        distance=distance*u.Mpc,
        inclination=inclination*u.rad,
        phi_ref=(phi_ref)*u.rad,
        f22_start=waveform_arguments['f_min']*u.Hz,
        f22_ref=waveform_arguments['reference_frequency']*u.Hz,
        f_max = max(freqs)*u.Hz,
        deltaF=(freqs[1]-freqs[0])*u.Hz,
    )
    return {'plus': hp.data, 'cross': hc.data}
```
</details>


- Wrapper file exposed to bilby. 
- Ideally store in the same location as the previous`gwsurr.py`
- Functions in this file:
    1. `parameter_conversion` mostly a convention thing
    2. `get_waveform_generator` wrapper to deal with parallel-bilby's occasional insolence 
    3. `NRHybSur3dq8_wrapper` function exposed to bilby (**do not change input/output format!**)

# GW150914.ini
<details>
<summary> Show Code </summary>

```bash
################################################################################
## Data generation arguments
################################################################################

trigger-time=1126259462.391

################################################################################
## Detector arguments
################################################################################

detectors = [H1, L1]
psd_dict = {H1=psd_data/h1_psd.txt, L1=psd_data/l1_psd.txt}
maximum-frequency={ 'H1': 896, 'L1': 896,  }
minimum-frequency={ 'H1': 20, 'L1': 20,  }
channel_dict = {H1:GWOSC, L1:GWOSC}
duration = 4

################################################################################
## Job submission arguments
################################################################################

label = GW150914
outdir = outdir

################################################################################
## Likelihood arguments
################################################################################

distance-marginalization=True
phase-marginalization=False
time-marginalization=True
jitter-time=True
reference-frame=H1L1
time-reference=geocent

################################################################################
## Prior arguments
################################################################################

prior-dict={
  chirp-mass: bilby.gw.prior.UniformInComponentsChirpMass(minimum=21.418182160215295, maximum=41.97447913941358, name='chirp_mass', boundary=None),
  mass-ratio: bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.11, maximum=1.0, name='mass_ratio', latex_label='$q$', unit=None, boundary=None),
  mass-1: Constraint(minimum=15, maximum=60, name='mass_1', latex_label='$m_1$', unit=None),
  mass-2: Constraint(minimum=15, maximum=60, name='mass_2', latex_label='$m_2$', unit=None),
  chi-1: Uniform(minimum=-0.91, maximum=0.91, name='chi_1', latex_label='$\chi_1$', unit=None, boundary=None),
  chi-2: Uniform(minimum=-0.91, maximum=0.91, name='chi_2', latex_label='$\chi_2$', unit=None, boundary=None),
  luminosity-distance: PowerLaw(alpha=2, minimum=10, maximum=10000, name='luminosity_distance', latex_label='$d_L$', unit='Mpc', boundary=None),
  theta-jn: Sine(minimum=0, maximum=3.141592653589793, name='theta_jn'),
  psi: Uniform(minimum=0, maximum=3.141592653589793, name='psi', boundary='periodic'),
  phase: Uniform(minimum=0, maximum=6.283185307179586, name='phase', boundary='periodic'),
  dec: Cosine(name='dec'),
  ra: Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic')
}
enforce-signal-duration=True
################################################################################
## Waveform arguments
################################################################################

waveform-generator=gwsurr_wrappers.get_waveform_generator
waveform-approximant=NRSurr
frequency-domain-source-model=gwsurr_wrappers.NRHybSur3dq8_wrapper
reference-frequency=20. 
waveform-arguments-dict ={
  f_min:20.,
}
conversion-function=gwsurr_wrappers.parameter_conversion
###############################################################################
## Sampler settings
################################################################################

sampler = dynesty
nact = 5
nlive = 1000
dynesty-sample = rwalk

################################################################################
## Slurm Settings
################################################################################

nodes = 1
ntasks-per-node = 128
time = 24:00:00
n-check-point = 10000
```
</details>

- the `Waveform Arguments` section is the place to make changes

```bash
waveform-generator=gwsurr_wrappers.get_waveform_generator
waveform-approximant=NRSurr
frequency-domain-source-model=gwsurr_wrappers.NRHybSur3dq8_wrapper
reference-frequency:20., 
waveform-arguments-dict ={
  f_min:20.,
}
conversion-function=gwsurr_wrappers.parameter_conversion
```
for some reason the code doesn't properly assign the conversion function, there is a check inside `get_waveform_generator` for this; to use a different approximant, make a new class in `gwsurr.py` and `gwsurr_wrappers.py` and specify it here under `frequency-domain-source-model`

- create `psd_data/`; add `h1_psd.txt` and `l1_psd.txt`; [see here](https://git.ligo.org/lscsoft/parallel_bilby/-/tree/master/examples/GW150914/psd_data)
- generate submission files with:
```bash
parallel_bilby_generation GW150914.ini
```
this will create an `outdir/` with submission and other files
# job file (SLURM, Unity)
inside `outdir/submit/analysis_GW150914_0.sh` replace the lines above the `mpirun` command with:
<details> 
<summary> Show Code </summary>

```bash
#!/bin/bash
#SBATCH --job-name=0_GW150914
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=24:00:00
#SBATCH --output=outdir/log_data_analysis/0_GW150914_%j.log
#SBATCH -p cpu-preempt


source ~/miniforge3/etc/profile.d/conda.sh
conda activate igwn-py310
```
</details>

then submit job with:
```bash
sbatch outdir/submit/analysis_GW150914_0.sh
```

# pesummary
when the pe run is complete, run in same directory as the .ini file:
```bash
summarypages -a NRHybSur3dq8 --email vupadhyaya@umassd.edu --webdir pesummary --samples outdir/results/GW150914_0_result.json --labels NRHybSur3dq8 --gw
```

put the results out for the world to see, eg.
```bash
cd public/gw/parallel-bilby_pe/
mkdir GW150914_gwsurrogate_new
cd GW150914_gwsurrogate
scp vupadhyaya_umassd_edu@unity.rc.umass.edu:/home/vupadhyaya_umassd_edu/surrogate_modeling/AlignedSpin/marginalization/parallel_bilby_pe/l_GW150914_gwsurrogate/pesummary .
```
