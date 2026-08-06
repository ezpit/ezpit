# This is for tutorial

Necessary Prerequisites Before Executing Python Code!!!
=======================================================

Installing Python and tools for writing codes
https://algotom.readthedocs.io/en/latest/toc/section1/section1_1.html

Setting up a Python workspace
https://algotom.readthedocs.io/en/latest/toc/section4/section4_1.html

***** To obtain I(q), S(q), F(q), G(r) from input xyz coordinates, or to derive S(q), F(q), G(r) from experimental I(q) 
or Compton scattering patterns, please refer to the files located in the """examples""" directory. ******
Sooner or later, we will incorporate various fitting functions to assess input models using experimental data.


Input file
==========

EZPIT is a software tool that can process ".xyz coordinate file" data from 
various 3D modeling programs. These programs include Mercury, Discovery Studio, 
VESTA, Avogadro, CrystalMaker and more. To use EZPIT with atom and ion, the user must 
match their name with the one in the atomic form factor file (aff_elementonly.txt). 
This file can be found in the "data" folder of EZPIT. Here are some examples of xyz 
coordinate file format.

![img_1.png](img_1.png)

An experimental G(r) file is a text file that contains the radial distribution 
function of a material. The file has two columns: the first one is the distance r 
in angstroms, and the second one is the value of G(r) at that distance. 
The file does not have any header or footer lines, and the columns are separated by 
spaces or tabs. An example of an experimental G(r) file is:

![img_2.png](img_2.png)

When you use xPDFsuite or pdggetX2 to produce G(r), you may notice some extra lines 
at the beginning of the file. These are header or footer lines that contain some metadata. 
If you want to keep them, you can specify the number of lines to skip in the row parameter 
in “def load_atom_names(file_path)” function.
EZPIT also requires two files including “aff_elementonly.txt” and “aff_parmonly.txt”. 
“aff_elementonly.txt” include all atoms and ions fore atomic form factors and “aff_parmonly.txt” 
has parameters for atomic form factors. The table of atomic form factor was obtained from 
“New Analytical Scattering Factor Functions for Free Atoms and Ions for Free Atoms and Ions by D. Waasmaier & A. Kirfel, 
Acta Cryst. (1995). A51, 416-413 and PDFgetX2 library”. 
Also International Tables for Crystallography (2006). Vol. C, ch. 4.3, pp. 259-429” provides scattering form factor. 
The equation of atomic form factor is 

![img_3.png](img_3.png)

where f(k) is the atomic form factor of the i-th atom or ion, q is the momentum transfer, 
ai, bi and C are the parameters from the table, and the summation is over four terms.

When the code reads each atom or ions in ".xyz coordinate file", the code finds the row of
each atom or ions in “aff_elementonly.txt” and picks the corresponding parameters 
such as ai, bi and C in the “aff_parmonly.txt”. 

![img_4.png](img_4.png)


Setup of input file path
========================

As an example, please specify the file path of experimental G(r), .xyz coordinate file, 
aff_elementonly.txt, and aff_paramonly.txt.

![img_5.png](img_5.png)

User input information
======================

The following parameters are required for getting scattering intensity I(q), 
structure function S(q), reduced structure function F(q), pair distribution 
function G(r): qmin, qmax, qstep, rmin, rmax, rstep, and qdamp.

I(q) calculation
================

We applied the Debye formula to calculate the scattering intensity I(q) from 
the atomic positions in the input .xyz coordinate file.

![img_6.png](img_6.png)

The total coherent scattering intensity, I(q), is theoretically defined as the sum of two fundamental components: 
self-scattering (i = j) and distinct-pair scattering (i ≠ j), a relationship clearly illustrated by the provided image using a three-element system (A, B, C).

![I(q) calculation_Basic equation_example.png](I%28q%29%20calculation_Basic%20equation_example.png)

If sample is water (H2O,  H1, H2, O1), I(q) = (fH1H1 + fH2H2 + fO1O1)2 = (2fH2 + fO2)
When q = zero (0), sin(qrij)/(qrij) = 1, So, the Debye formula will be:

![img_8.png](img_8.png)

If sample is water (H2O,  H1, H2, O1), I(q) = (fH1 + fH2 + fO1)2 = (2fH + fO)2 
One possible way to explain this is:
The .xyz coordinate file contains the positions of the atoms in a molecule. 
For example, if the .xyz coordinate file has 

![img_9.png](img_9.png)

It means that there are three atoms: one iridium and two oxygen. 
To calculate the scattering intensity I(q) as a function of the momentum transfer q, 
we need to consider the distances between all pairs of atoms. 
At q = zero (0), 2nd term is only considered for I(q) When q > 0, 
all pairs are considered as a matrix like below. Diagonal components (when I = j) 
will be  2nd term which is same as when q=zero.

![img_10.png](img_10.png)

To speed up the computation of I(q) and avoid redundant calculations, 
we used a matrix approach. The function "def group_atoms(atom_names)" 
allows us to extract various information from the .xyz coordinate file, 
such as the names of the atoms, their number, their index, 
and their diversity.

![img_11.png](img_11.png)

S(q) calculation
================

To determine the structure function S(q), we applied the following formula.

![img_12.png](img_12.png)

The function “calculate_Sq” takes several arguments, such as atom_indices, 
scattering_factors, atom_distance_matrix, qmin, qmax, qstep, and return_Iq. 
It computes the structure function S(q) for a given set of atoms and 
scattering factors. And the mean of square atomic form factor <f2> 
and the square of mean atomic form factor <f>2 can also be calculated.

G(r) calculation by integral function and inverse fast Fourier transform (IFFT)
===============================================================================

The integral calculation method is more time-consuming than the IFFT method 
because it requires repeated calculations from rmin to rmax . For each r value, 
the function G(r) has to be integrated from qmin to qmax.

![img_13.png](img_13.png)

To avoid repletion of calculation, a list was used and please see the function of 
“calculate_Gr_integral”. The following figure illustrates the quality of 
PDF spectrum with different rsetp.

![img_14.png](img_14.png)

The IFFT method is more efficient than the integral method, but it needs 
the relation between q (reciprocal space) and r (real space) to obtain 
a correct real-space value for G(r). After applying IFFT, the imaginary part of G(r)
was extracted. In addition, the real-space axis had to be adjusted with

total_point = int(2 * 3.14159 / (rstep * qstep))
rfine = np.arange(total_point) * rstep

In order to perform an inverse fast Fourier transform (IFFT), 
the data does not have to be a power of 2 (2N) anymore. This means that 
the total number of data points, including the ones that are padded with zeros, 
can be any positive integer. However, if the function F(q) does not start from zero, 
then it is necessary to add a zero value (or an extrapolated or interpolated value) 
for F(q) between zero and qmin before applying the IFFT.


![img_15.png](img_15.png)

The following figure illustrates how to pad the data by interpolation and 
extrapolation when the line below qmin is zero. This technique helps to
avoid artifacts and improve the quality of the inversion results. 
The padding is done by fitting a straight line (y = ax or y=ax +b) to 
the data points and extending it beyond the original range of qmin and qmax.
In this code, “interpolate.interp1d (extrapolation=orangeline)” is used fill gap between qmin and zero.

![img_16.png](img_16.png)

The following paragraph shows how to improve the resolution of G(r) 
by using zeropad in F(q). Zero padding is a technique that inserts zeros 
after qmax in F(q) before performing IFFT. This increases the number 
of data points in G(r) and makes the peaks smoother. Zero padding is a technique 
that improves the quality of G(r) by inserting additional data points 
between the original ones obtained from F(q). This does not mean adding zeros, 
but rather interpolating the data to increase the resolution. 
As a result, the G(r) curve is smoother and more accurate. Without zeropad, 
G(r) has fewer data points and the peaks are more jagged. 
Here is an example of the difference between using zeropad and 
not using zeropad in F(q) and G(r).

![img_25.png](img_25.png)

Example:
========
With the xyz coordinate file, EZPIT provides I(q) to G(r). qdamp was zero (0).

![img_19.png](img_19.png)

No ions in Ni(OH)2-109391-ICSD-10x10x1.xyz. please see the doc directory.

![img_21.png](img_21.png)

Ions in 5IrC_r5a-1Ir_ion.xyz

![img_22.png](img_22.png)


Calculating Compton scattering pattern
======================================

The theoretical Compton scattering intensity, Iinc(q) (Egami & Billinge, 2003), shown in equation below, 
represents the incoherent scattering contribution arising from the inelastic collision between X-ray photons and electrons. 
Unlike coherent scattering, this component carries no structural information and acts as a background that increases with q, 
significantly affecting the normalization of S(q) at high angles. EZPDF evaluates the incoherent intensity per atom 
using the analytic parametrization of Balyuzi (Balyuzi, 1975), who fitted the incoherent scattering factors of Cromer & Mann 
(Cromer & Mann, 1967) and Cromer (Cromer, 1969) to a five-Gaussian function of s = sin(θ)/λ. 
For each element a, this parametrization yields the quantity fₐ(q) = Za − Iinc(q), 
which approaches the atomic number Za at low q and decays toward zero at high q. 
Because Balyuzi fitted the quantity Za − Iinc(q) directly, the incoherent contribution of atom a is recovered as Za − fa(q), 
which approaches zero at low q and Za at high q. The total Compton intensity is then the concentration-weighted sum over all atoms, 
scaled by the Breit–Dirac recoil factor of equation (11), as given in equation below.

![img_26.png](img_26.png)

ca (atomic fraction) = Na/N, where Na is the number of atoms of type a and N is the total number of atoms in the chemical unit); 
Za is the atomic number of the ath type of atom; and fa(q) is the analytic incoherent-scattering parametrization of Balyuzi (1975). 
f(q) = Ffit(q) = Za – Iainc(q).
The Compton (incoherent) scattering coefficients were obtained from D. T. Cromer, J. Chem. Phys. 50, 4857 (1969), 
as parametrized by H. H. M. Balyuzi, Acta Cryst. A31, 600 (1975). Balyuzi fitted the function F_fit(s) = Z − I_inc(s) 
(not the incoherent intensity itself) to a sum of five Gaussians, where Z is the atomic number and s = sin(theta)/lambda. 
The fitted function is:

![img_27.png](img_27.png)

where F_fit(k) is Balyuzi's five-Gaussian fit for the i-th atom (no ion information). Note that F_fit is NOT an atomic form factor: 
it equals Z − I_inc, so the incoherent (Compton) intensity per atom is recovered by simple subtraction, I_inc = Z − F_fit. 
k is converted to momentum transfer (q) using (0.25⋅q/π)**2, where ai, bi, and C are the parameters from  "compton_elementonly.txt". 
The summation is over five terms. When the code reads each atom in the ".xyz coordinate file," it identifies the row of each atom in 
"compton_elementonly.txt" and extracts the corresponding parameters such as ai, bi, and C from "compton_parmonly.txt."

![img_28.png](img_28.png)

The code that evaluates Balyuzi's fitted function F_fit(k) is shown below. 
The incoherent intensity is then obtained as I_inc = Z − F_fit (averaged per atom and multiplied by the Breit–Dirac recoil factor).

![img_29.png](img_29.png)

The necessary information for calculating the Compton scattering pattern is

![img_30.png](img_30.png)

The plot of the calculated Compton scattering pattern for the composition Co2 O2 P1 is displayed below.

![img_31.png](img_31.png)

Also, several important functions in "loadsaver.py" are explained in terms of their roles. Please see the examples below.

![img_32.png](img_32.png)

![img_33.png](img_33.png)


Calculating S(q), F(q), G(r) from experimental I(q)
===================================================
Figure displays the experimental intensity I(q) (denoted as Exp(Raw), the scaled background (represented as Bkg*Scale), 
the background-subtracted experimental intensity (Net I(q))

![img_36 .png](img_36.png)

Figure presents the background-subtracted experimental intensity (Net I(q))

![img_37.png](img_37.png)

Figure exhibits the structure function (not normalized S(q) = Standard S(q)) derived from the experimental intensity I(q) 
(Net I(q) in the previous Figure. The not normalized S(q) shows an increase in intensity as a function of increasing q (1/Å), 
which is attributed to the imperfection of all subtraction and the instrumental limitations. 
To bring the oscillation of S(q) around 1, a form of correction is necessary.
EZPDF employs a polynomial correction, similar to the approach taken by xPDFsuite. The normalized S(q) (Poly-corrected S(q)) then oscillates around 1.

![img_38.png](img_38.png)

As an example of polynomial correction of S(q), polynomial order is used instead of r-poly, which is adopted in xPDFsuite.
The black line represents the "Experimental S(q)" which is not corrected by polynomial.
The red line denotes the calculated polynomial for the black line.
The blue line illustrates the S(q) after subtracting the red line from the black line.
To obtain a polynomial function, a Vandermonde matrix with a least squares fit is utilized here.
Reference materials:
https://m.youtube.com/watch?v=BExDXaFOjF4
Prof. Wen Shen, Penn State Univ.
https://www.youtube.com/watch?v=G6lVSD0Jci0&t=211s
Numerical recipes by W. Press, B. Flannery, S. Teukolsky, W. Vetterling

![img_38-1.png](img_38-1.png)


More detail of S(q) calculation
===============================
Due to imperfection of all subtraction, the calculated S(q) does not oscillate about the baseline (1), shown as baseline (1) shown in the black line. 
So, To bring oscillation of S(q) to 1 shown in redline (Calculated polynomial), we need to use polynomial correction. 
Vandermonde matrix and least square fit are combined to provide polynomial equation. 
Actually Numpy _polyfit_ in python use the same method. So, you can used Numpy polyfit directrly.
The experimental structure function S(q) is obtained from the background-subtracted intensity using an ad hoc X-ray normalization 
(PDFgetX3 by Juhás et al., J. Appl. Cryst. 46, 560 (2013)).

![Experimental S(q) calculation.png](Experimental%20S%28q%29%20calculation.png)

As an illustrative example, data from disordered cobalt oxide is presented. 
The top panel displays the scattering intensities for the cobalt oxide sample(black), the background (red), and the resulting background-subtracted data(blue). 
I(q) is plotted on a logarithmic scale, which provides a much clearer visualization of the subtracted data compared to a linear scale.
The middle panel shows the structure function, S(q), calculated from the background-subtracted data. 
This particular result arises because unwanted signals were not perfectly eliminated. By applying a polynomial correction to the calculated S(q), 
the peaks are adjusted to oscillate around unity (1). The low panel demonstrates how polynomials of two different degrees fit the calculated S(q) data. 
For better visualization, these polynomials were shifted by an offset of 1 and superimposed onto the S(q) plot. 
Notably, the high-degree polynomial (order 20.0, blue) exhibits significant deviations from the appropriate degree (order 7.208, red), particularly in the low-q region.  
EZPDF constructs the polynomial fit explicitly from the Vandermonde matrix and numpy.linalg.lstsq rather than calling numpy.polyfit; the two are mathematically equivalent, 
but the explicit formulation gives direct control over the design matrix used for the non-integer-order interpolation described below. 
This error component is subsequently subtracted from the raw data to yield the final structure factor. 
To provide finer control over this normalization, the software extends this approach to support non-integer polynomial orders through a linear interpolation strategy. 
When a real number is specified as the target polynomial order (poly_order), the system first calculates the correction functions for the nearest lower integer order (poly_lo) 
and the nearest higher integer order (poly_hi) using the Vandermonde-based least-squares method described above. 
Subsequently, weights are calculated to determine the contribution of each boundary order to the final result, 
where the fractional part of the target order determines the weight for the higher order (w_hi = poly_order – lo), 
and the remaining value determines the weight for the lower order (w_lo = 1 – w_hi). 
The final continuous correction function corresponding to the real-valued order is yielded by linearly combining the results 
from the two integer orders according to the formula P(q)Polynomial_for_sq = w_lo × P(q)poly_lo + w_hi × P(q)poly_hi. 
For example, when utilizing a non-integer target polynomial order of 7.208, the interpolation is based on the closest integer boundaries of 7.0 and 8.0. 
The weight assigned to the higher order (w_hi) is the fractional part (0.208), and the remaining weight for the lower order (w_lo) is 0.792. 
Consequently, the final interpolated polynomial correction function is derived as a linear combination given by P(q)7.208 = 0.792 × P(q)7 + 0.208 × P(q)8.


![S(q)_calculation_polynomials.png](S(q)_calculation_polynomials.png)

The top graph in the second set depicts the data length required before performing an Inverse Fast Fourier Transform (IFFT), 
which requires padding data from 0 to qmin (red) and after qmax (blue). 
Although an integral function can calculate the data from qmin to qmax without padding, the process is much slower than the IFFT method.
The second, third, and bottom graphs evaluate how various padding methods after qmax influence the resulting G(r), 
showing no significant difference between Zero, Constant, and Decay padding for distances above 1 Å. 
Consequently, the EZPDF GUI version uses Zero padding for any data added after qmax.


![Pad_F(q)_Gr result.png](Pad_F(q)_Gr result.png)

How  polynomial subtraction in S(q) affect on G(r):
The  top figure display how two different polynomial orders are superimposed on S(q). 
The bottom plot illustrates the pair distribution function G(r), revealing that using a high-order polynomial leads to a loss of peaks in the low-r region(red). 
This occurs because the high-order polynomial subtracts an excessive amount of intensity from the original calculated S(q).

![Polynomial_low_r_Gr.png](Polynomial_low_r_Gr.png)

For comparison, S(q) (xpdfsute_S(q)) obtained from xPDFsuite is also displayed. 
Both S(q) representations are quite similar except low q in S(q).
the middle figure presents the reduced structure function (F(q)) derived from S(q) in Figure d), where F(q) oscillates around 0. 
For comparison, F(q) (xpdfsute_F(q)) from xPDFsuite is also shown. Apart from a difference in scaling, 
both versions of F(q) are identical.
The bottom figure displays pair distribution functions (integral-G(r), ifft-G(r)) derived from F(q) in the middle figure using inverse fast Fourier transform (ifft). 
For comparison, G(r) (xpdfsute_G(r)) from xPDFsuite is also shown. the pair distribution functions are identical.

![img_s(q)_f(q)_g(r)_comparison.png](img_s%28q%29_f%28q%29_g%28r%29_comparison.png)


GUI of EZPDF 
=============
The picture shows GUI of EZPDF to process experimental I(q) data to obtain S(q), F(q), G(r). the loaded file, the necessary parameters, 
and the smoothing function are shown. Also, “Open in New Graphs” allows new graph when open each file.
Load file, folder, saved project file which as parameters and file path. View tab show “Dark Mode”.
Select several files and delete all. **File drag-and-drop and deleted file recovery (Undo Delete) functions have been added.** 
EZPDF panel

![EZPDF_GUI.png](EZPDF_GUI.png)

Under “File”, “Open” each file, or selected files or files in folder can be loaded.
“Opne Project” A saved project file” can be loaded.
“Save Project” The parameters of your current working files can be saved as a project file

![EZPDF_panel_fileload.png](EZPDF_panel_fileload.png)

Under “View”, Changes the GUI theme from Light to Dark Mode.

![EZPDF_panel_Darkmode.png](EZPDF_panel_Darkmode.png)

Clicking the icon launches the EZPDF plot window as a pop-up

![EZPDF_panel_2Dgraph_icon.png](EZPDF_panel_2Dgraph_icon.png)

The picture shows the EZPDF Plot, displaying I(q) (black) including background data (red), and background subtracted data (blue), 
S(q) with calculated S(q)(black) including polynomial data (red) and polynomial corrected S(q)(blue), F(q) (blue) and smoothed F(q) (black), 
and G(r)(blue) and G(r) (black) from smoothed F(q).
“Lock graph” lock the parameter on the graph. All parameter in “Lock graph” will not be change even though parameters is changed in EZPDF parameter. 
So. You compare data with different data which have different parameters.
Toggles (I(q), S(q), F(q), G(r)) shown in the down right corner) shows specific data only. 

![EZPDF_plot.png](EZPDF_plot.png)

The picture shows “Dark Mode” of EZPDF and EZPDF Plot.
![EZPDF__EZPDF_plot_Darkmode.png](EZPDF__EZPDF_plot_Darkmode.png)

“Save data” saves processed data for experimental data or calculated data from input xyz model data. 
“Save figure” (not shown) saves EZPDDF Plot as PNG, JPEG, SVG, PDF.

![EZPDF_plot_save_plot option.png](EZPDF_plot_save_plot%20option.png)

EZPDF plot window: multiple plot: 
X Offset and Y Offset are available.
Selective plotting is also available.

![EZPDF_plot_offset_miltiplot.png](EZPDF_plot_offset_miltiplot.png)

Clicking the 'Delete Selected' button will trigger a 'Confirm Deletion' prompt. You can then select either 'Yes' to proceed or 'No' to cancel.

![Delete Selected.png](Delete%20Selected.png)


Prerequisites
=============
Python 3.11: The application utilizes syntax and features introduced in Python 3.11.
Note: EZPDF has been specifically validated on 3.11; compatibility with newer versions (3.12+) is currently untested.
Operating System: Windows (64-bit).
Note: 32-bit environments, macOS, and Linux are currently untested.

Dependencies
Core Framework: PySide6 is the primary dependency for the GUI.
Compatibility Check: Before installation, please ensure your Python environment aligns with PySide6 version requirements to prevent library conflicts.
![Reqired software1.png](Reqired%20software1.png)
![Reqired software2.png](Reqired%20software2.png)
![Reqired software3.png](Reqired%20software3.png)


Smooth function: WH_smooth_GUI
==============================
A common challenge in pair distribution function (PDF) analysis arises when the scattering signal of interest is weak relative to, 
or comparable in magnitude with, the background, as is typical for low-concentration samples, thin films, dilute solutions, 
in situ cell measurements, or operando electrochemical setups. In such cases, even small inaccuracies in the background measurement or 
in the determination of the background scaling factor α can leave residual sharp features, glitches, 
or oscillatory artifacts in the background-subtracted intensity I(q) − α × IBkg(q). These residuals are often confined to narrow q intervals but, 
if left untreated, propagate into the Fourier transform and contaminate the resulting G(r) with spurious peaks or low-r ripples. 
To address this issue without distorting the genuine scattering signal elsewhere, a dedicated range-selective processing utility, 
WH_smooth_GUI, has been developed as a standalone companion tool to EZPDF. The four available modes are: (i) Whittaker–Henderson (WH) smoothing, 
which suppresses high-frequency noise while preserving the underlying scattering envelope; (ii) Linear interpolation, which replaces a q interval with
a straight line between its boundary points and is well-suited for removing very narrow, isolated artifacts; (iii) Automatic cubic-spline interpolation, 
which fits a natural cubic spline through 2N anchor points sampled symmetrically outside the targeted q interval, providing a smooth baseline restoration 
that follows the local trend of the data; and (iv) Manual cubic-spline interpolation, in which the user directly clicks on the graph to define anchor points, 
giving full control when the residual feature is irregular or when the automatic methods fail to capture the desired baseline shape. 
All spline-based modes are performed in log-space when the data are strictly positive, so the reconstructed segment correctly follows the exponentially 
decaying nature of typical scattering profiles. Independent λ, polynomial order, and blend-width parameters can be assigned to every q range, 
allowing the processing strength and the cosine-taper transition width to be tailored to the local character of the data. At each range boundary, 
a cosine-taper blending function ensures a smooth, artifact-free transition between the processed segment and the original data, preventing the introduction of discontinuities.
Figure  shows the GUI of the standalone WH Range Smoother utility applied to an experimental dataset. The tool performs processing on the background-subtracted intensity I(q) − α × IBkg(q), 
using a scale factor α = 2.17 to balance the sample and background contributions. WH smoothing is applied independently across three user-defined q ranges: 
q = 1.9 – 2.3 / Å with λ = 10,000,000 (blend = 14 points), q = 2.3 – 8.0 / Å with λ = 1,000 (blend = 20), and q = 10 – 25 / Å with λ = 1,000 (blend = 20). 
The cosine-taper blend at each boundary preserves continuity with the unmodified regions of the data, as confirmed by the residual curve in the lower panel, 
where ΔI ≈ 0 outside the shaded smoothing windows. The use of a per-range λ parameter is essential because the local character of the scattering signal differs significantly across the q axis: 
at low q (1.9 – 2.3 / Å) the data contain sharp, narrow features that require very strong smoothing to suppress, whereas at higher q a moderate λ is sufficient 
to remove statistical noise without distorting the underlying scattering envelope. The reference curve (scaled by a factor of 10) overlaid on the main panel serves 
as an independent validation that the smoothed result faithfully reproduces the expected scattering profile across the entire q range.

Graphical user interface of the standalone WH Range Smoother utility. (Top) Overlaid curves on a logarithmic I(q) scale: raw sample intensity I(q) (black), 
scaled background α × IBkg(q) (blue), background-subtracted intensity I(q) − α × IBkg(q) (red), the WH smoothed result (purple), and a scaled reference curve 
(green dashed) for comparison. Yellow shaded bands indicate the user-defined q ranges over which smoothing is applied. 
The inset (green rectangle) expands the q = 1.9–2.3 / Å region. (Bottom) Difference curve ΔI(q) = I(q) – ISmoothed(q).

![WH_smooth_GUI-5.jpg](WH_smooth_GUI-5.jpg)
