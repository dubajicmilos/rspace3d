"""
help_content.py - The text of the help pages of the rspace3d windows.

`TOPICS` is the ordered list of (key, title, html). The text is the simple
HTML that a QTextBrowser shows. A link to another topic is
`<a href="topic:KEY">`. `help_view.py` shows the pages and draws the figures
(`<img src="fig:NAME">`).

The definitions that the program also prints (split-half difference, frame
map, limits of the score) come from `rspace3d.averaging`, so the help and the
program use the same words.
"""
from __future__ import annotations

import html

from . import averaging as av
from .volume_builder import LAUE_GROUP_NAMES, METRIC_TOLERANCE, _EXPECTED_ORDERS

EXPLORER = 'explorer'       # the topic that `help_view` shows as an interactive page
STD_RAW = av.COUNTS_MODEL['name']
STD_RAW_FALLBACK = av.COUNTS_MAD['name']
STD_UNWARP = av.EQUAL_MAD['name']
REFERENCE = av.BIAS_REFERENCE


def _table(head: list[str], rows: list[list[str]], widths: list[int] | None = None) -> str:
    cols = ''
    if widths:
        cols = ''.join(f'<col width="{w}%">' for w in widths)
    out = [f'<table class="grid" cellspacing="0" cellpadding="6" width="100%">{cols}<tr>']
    out += [f'<th align="left">{h}</th>' for h in head]
    out.append('</tr>')
    for r in rows:
        out.append('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>')
    out.append('</table>')
    return ''.join(out)


def _note(text: str, kind: str = 'info') -> str:
    colour = {'info': '#E3EEF9', 'warn': '#FFF1D6', 'ok': '#E2F3E9'}[kind]
    return (f'<table cellspacing="0" cellpadding="10" width="100%"><tr>'
            f'<td bgcolor="{colour}">{text}</td></tr></table>')


def _laue_rows() -> list[list[str]]:
    return [[html.escape(key), html.escape(name), str(_EXPECTED_ORDERS[key])]
            for key, name in LAUE_GROUP_NAMES.items()]


START = f'''
<h1>Start here</h1>
<p>rspace3d makes a 3D volume of reciprocal space from single-crystal X-ray scattering data.
It averages the volume over the symmetry of the crystal, and it shows planes of the volume.</p>

<h2>Two routes to a volume</h2>
{_table(['Route', 'Input', 'Window', 'Command'], [
    ['A. Raw frames',
     'The raw <code>.cbf</code> frames, the <code>*_cracker.par</code> and the '
     '<code>*_peakhunt.tabbin</code> of the CrysAlisPro experiment',
     'Reconstruction from raw frames', '<code>rspace3d-reconstruct</code>'],
    ['B. Unwarp layers',
     'The <code>.img</code> layers that CrysAlisPro writes with <code>dc unwarp</code>',
     'Volume from unwarp layers', '<code>rspace3d-builder</code>'],
], [16, 44, 22, 18])}
<p>The two routes write the same HDF5 file format. They use the same binning, the same
symmetry average and the same viewer (<code>rspace3d-viewer</code>).</p>

<h2>A standard run</h2>
<ol>
<li>Select the folder with the data.</li>
<li>Check the Laue group. In route A also check the grid, in route B the binning.</li>
<li>Keep the standard averaging. For data with contamination, compare the options on the
Averaging page first.</li>
<li>Click the main button (<b>Reconstruct and save</b> or <b>Process all</b>). The log at the
bottom of the window shows each step.</li>
<li>Open the result in the viewer.</li>
</ol>

<h2>Topics</h2>
<ul>
<li><a href="topic:raw-frames">Reconstruct from raw frames</a> (route A, step by step)</li>
<li><a href="topic:unwarp">Volume from CrysAlisPro layers</a> (route B, step by step)</li>
<li><a href="topic:symmetry">Symmetry average and Laue group</a>: what the average does to
the data</li>
<li><a href="topic:averaging">Averaging options</a>: weights, outlier rejection, threshold</li>
<li><a href="topic:{EXPLORER}">Try the options on one voxel</a> (interactive example)</li>
<li><a href="topic:averaging-page">Compare the options on your data</a> (the Averaging page)</li>
<li><a href="topic:frame-tools">Frame tools</a> (powder rings and other contamination that
changes between frames)</li>
<li><a href="topic:grid">Grid, voxel size, binning and memory</a></li>
<li><a href="topic:advanced">Advanced settings</a></li>
<li><a href="topic:viewer">Look at a volume</a> (the viewer)</li>
<li><a href="topic:view3d">Look at a volume in 3D</a> (surfaces of equal intensity,
projection)</li>
<li><a href="topic:files">Output files</a></li>
<li><a href="topic:glossary">Glossary</a></li>
</ul>

<h2>Keys</h2>
<p><b>F1</b> opens the help for the page that you see. <b>Ctrl+Return</b> starts
<i>Reconstruct and save</i> on the Reconstruct page, or <i>Process all</i> on the Process
page. A small <b>?</b> button opens the help for one part of a page.</p>
'''

RAW_FRAMES = f'''
<h1>Reconstruct from raw frames</h1>
<p>This window (route A) calculates the volume from the raw detector frames. A
<code>dc unwarp</code> step in CrysAlisPro is not necessary.</p>

<h2>What you need</h2>
<p>One folder, normally the CrysAlisPro experiment folder, with these files:</p>
{_table(['File', 'Purpose'], [
    ['<code>&lt;stem&gt;_0001.cbf</code>, <code>&lt;stem&gt;_0002.cbf</code>, ...',
     'The raw rotation frames. The window reads the intensities, the wavelength, the pixel '
     'size, the distance, the beam centre and the rotation angles from them.'],
    ['<code>&lt;exp&gt;_cracker.par</code>',
     'The refined UB matrix (unit cell and orientation) and the wavelength.'],
    ['<code>&lt;exp&gt;_peakhunt.tabbin</code>',
     'The peak table of CrysAlisPro. The geometry fit uses it.'],
], [36, 64])}
<p>CrysAlisPro writes the <code>.par</code> and the <code>.tabbin</code> when you run the
peak hunt and then find and refine the unit cell. Run the unit-cell step after the peak
hunt. Use the <code>_cracker.par</code>, not the plain <code>&lt;exp&gt;_0.par</code>.</p>

<h2>The steps of the Reconstruct page</h2>
<ol>
<li><b>Dataset.</b> Click <b>Browse</b> and select the folder. The window shows the number
of frames, the detector, the wavelength, the scan range and the unit cell. If the window
finds no <code>.par</code> file, tick <i>Search subfolders for the .par file</i> on the
Advanced page.</li>
<li><b>Grid.</b> Set the range of h, k and l and the voxel size
(<a href="topic:grid">more</a>). The column <i>data reach</i> shows how far the data go.
The line <i>Grid size</i> shows the number of voxels and the GPU memory that the
reconstruction needs.</li>
<li><b>Symmetry.</b> Select the Laue group of the phase
(<a href="topic:symmetry">more</a>).</li>
<li><b>Averaging and outlier rejection.</b> The line shows the option that the symmetry
average uses. The standard for raw frames is <i>{STD_RAW}</i>
(<a href="topic:averaging">what the options do</a>).</li>
<li><b>Output.</b> The folder (you can change it), the tick for the raw volume and the
names of the files (<a href="topic:files">more</a>). The window asks before it overwrites a
file.</li>
</ol>
<p>Then click <b>Reconstruct and save</b>. One click reconstructs, averages and saves. The
log shows each step.</p>
{_note('If you change only the Laue group, the averaging or the output, the next click uses '
       'the reconstruction in memory. It does not reconstruct again. A change of the grid, the '
       'frames, the bin factors, the pixel corrections or the geometry option starts a new '
       'reconstruction.')}
<p><b>Cancel</b> stops a reconstruction after the current frame. The window keeps the earlier
reconstruction. <b>Open in viewer</b> opens the last saved symmetrised volume.</p>

<h2>Check the result in the log</h2>
<ul>
<li><i>best fit rms</i> is the residual of the geometry fit, in 1/&Aring;. Values near 0.001
to 0.002 are typical of a good fit.</li>
<li><i>validation: fitted peaks -&gt; integer hkl, median dev</i> is a few hundredths of an
r.l.u. for a good fit. The same line gives the value for the peaks that the fit did not use
(<i>held-out</i>). If one of the two values is above 0.1 r.l.u., the run stops.</li>
<li><i>Bragg registration RMS</i> is the distance of the 300 brightest voxels from the nearest
integer node. A value below the voxel size means that the volume agrees with the lattice.
Strong powder rings can make this number larger, so look at the volume too.</li>
<li><i>measured ... voxels</i> is the part of the grid that the scan covers.</li>
</ul>
<p>Then open the volume in the viewer, switch on the Miller-index grid and look at some
planes. The Bragg peaks must be on the crossings of the grid lines in all three plane
orientations.</p>

<h2>Problems</h2>
{_table(['Message or symptom', 'Cause and fix'], [
    ['<i>Found N .cbf frames but no *_cracker.par / *.par</i> (below the folder), or '
     '<code>no CrysAlisPro .par file found</code> (in the log)',
     'The <code>.par</code> is not in the selected folder. Tick <i>Search subfolders for the '
     '.par file</i> on the Advanced page.'],
    ['<code>need a *_cracker.par and a *_peakhunt.tabbin</code>',
     'The folder has no peak table. Run the peak hunt in CrysAlisPro.'],
    ['<code>only N cleanly-indexed tabbin peaks</code>',
     'The peak table does not agree with the UB matrix in the <code>.par</code>. Find and '
     'refine the unit cell again after the peak hunt.'],
    ['<code>tabbin geometry fit failed self-check</code>',
     'Check that the frames in the folder are the scan of the peak hunt. The scan must '
     'have one rotation axis and a flat detector.'],
    ['GPU out of memory',
     'Make the grid smaller (a larger voxel size or narrower ranges), or untick <i>Use the '
     'GPU</i> on the Advanced page.'],
    ['Peaks are smeared',
     'The geometry is wrong. Keep <i>Geometry from the CrysAlisPro peak table</i> ticked.'],
    ['No GPU',
     'The CPU gives the same result, at about 1 s for each frame on a 481 x 481 x 481 grid. '
     'Use fewer frames (Advanced page) or a larger voxel size for a first test.'],
], [38, 62])}
'''

UNWARP = f'''
<h1>Volume from CrysAlisPro layers</h1>
<p>In this route (route B) CrysAlisPro calculates the planes of reciprocal space. The window
puts them into one volume with the correct (h, k, l) grid, which it takes from the UB matrix
of the <code>.par</code> file.</p>

<h2>Step 1: the layer list</h2>
<p>CrysAlisPro makes one HK plane for each value of l with the command
<code>dc unwarp</code>. It needs a list of the layers. Open the <b>Layer list</b> page. Set
the range of l (for example -6 to 6), the step (for example 0.02) and the resolution limit
(for example 0.8 &Aring;). Then click <b>Generate the layer list</b>. CrysAlisPro accepts
500 layers in one list. A longer list goes into two or more files.</p>

<h2>Step 2: unwarp in CrysAlisPro</h2>
<p>In the command line of CrysAlisPro, run <code>dc unwarp</code> and load the
<code>.dcunwarp</code> file. Do this once for each file. CrysAlisPro writes numbered
<code>.img</code> files into a folder with the name <code>unwarp</code>.</p>

<h2>Step 3: the volume</h2>
<ol>
<li>On the <b>Process</b> page, select the <code>unwarp</code> folder. The window finds the
<code>*_cracker.par</code> (normally in the folder above). It gives the UB matrix for the
grid.</li>
<li>Select the Laue group (<a href="topic:symmetry">more</a>).</li>
<li>Set the binning. 2 x 2 in the plane is the recommended value.</li>
<li>Click <b>Process all</b>. The window writes <code>&lt;sample&gt;_raw.h5</code> and
<code>&lt;sample&gt;_sym_&lt;laue&gt;.h5</code> into the unwarp folder.</li>
</ol>
<p>The standard averaging for unwarp layers is <i>{STD_UNWARP}</i>. The layers have no
detector pixel counts. After a binning, the count of a voxel is only the number of its pixels
with a measurement. Thus the pixel-count weights of route A give almost no gain here
(<a href="topic:averaging">what the options do</a>).</p>
{_note('The window keeps the raw volume in <code>&lt;sample&gt;_raw.h5</code>. The next run '
       'reads this file. It does not read the <code>.img</code> files again if the mask '
       'kernel is the same and the <code>.img</code> and <code>.par</code> files did not '
       'change. Tick <i>Read the .img files again</i> on the Advanced page to read them again '
       'in each case.')}

<h2>Pixels without a measurement</h2>
<p>CrysAlisPro writes 0 for regions without a measurement (beam stop, gaps between detector
panels), but 0 is also a valid low count. The window finds the regions without a measurement
from their size and marks them as not measured. The symmetry average then does not use
them. The <i>Mask kernel</i> on the Advanced page sets the size that separates the two
cases.</p>
'''

SYMMETRY = f'''
<h1>Symmetry average and Laue group</h1>
<p>Each voxel of a single-crystal volume has symmetry-equivalent positions under the Laue
group of the diffraction pattern: up to 48 for the cubic group m-3m. This set of positions is
the <b>orbit</b> of the voxel, and each measured position is a <b>member</b> of the orbit.
For each voxel, the symmetry average collects the members and removes the members that are
far from the others. It then writes the mean of the members that remain.</p>
<p><img src="fig:orbit"></p>

<h2>What the average does to the data</h2>
<ul>
<li><b>It fills gaps.</b> A region that the scan did not reach gets its value from the
measured equivalent positions.</li>
<li><b>It removes intensity that does not follow the symmetry.</b> A zinger or a bright arc
is in one member only. The outlier rejection removes that member. A complete powder ring is
itself symmetric. It stays.</li>
<li><b>It lowers the noise.</b> Each voxel becomes the mean of some independent
measurements.</li>
</ul>
{_note('The average gives the data the symmetry that you select. Use the Laue group of the '
       'measured diffraction pattern. For a crystal with twins or domains, use the group of '
       'the pattern that the domains give together. Keep the raw volume for each analysis '
       'that looks for a break of the symmetry.', 'warn')}

<h2>Which Laue group</h2>
<p>Select the group of the phase. An experiment that lowers the symmetry (for example an
applied field) needs a lower group. A group that is too high mixes positions that are not
equivalent.</p>
<p>The windows make one check: the metric of the unit cell must allow the group (the
residual must not be more than {METRIC_TOLERANCE:g}). If the metric does not allow the
selected group when you load a folder, the window selects the allowed group of highest
order. It also shows a note. The metric cannot tell a cubic cell from a cell with a small
distortion, so the choice of the group stays yours.</p>
{_table(['Laue group', 'Name in the list', 'Operations'], _laue_rows(), [20, 55, 25])}
<p>The number of operations is the largest number of members of an orbit. A voxel on a mirror
plane or on a rotation axis has fewer different members.</p>
'''

AVERAGING = f'''
<h1>Averaging options</h1>
<p>The symmetry average makes one value from the members of an orbit
(<a href="topic:symmetry">what an orbit is</a>). An averaging option has three parts: the
weights of the members, the rule that rejects outliers, and the threshold of this rule. The
name of an option gives the three parts, for example <i>{STD_RAW}</i>.</p>
<p>To see the rules at work on one voxel, open
<a href="topic:{EXPLORER}">Try the options on one voxel</a>.</p>

<h2>1. Weights: how much each member counts</h2>
{_table(['Weights', 'What it does', 'Effect on the data', 'Use it for'], [
    ['<b>equal weights</b>',
     'The plain mean of the members that remain.',
     'A member from a few pixels counts as much as a member from many pixels.',
     'Unwarp layers (route B). They have no detector pixel counts.'],
    ['<b>pixel-count weights</b>',
     'Each member has the weight of its pixel count: the number of detector pixels that '
     'went into the voxel.',
     'A member from many pixels has less noise and counts more. The noise of the mean is '
     'lower where the pixel counts of the members are very different.',
     'Raw frames (route A). This is in the standard option.'],
    ['<b>noise weights</b>',
     'Each member has the weight 1 / variance. The variance comes from a noise model that '
     'the program fits to the volume.',
     'Close to the pixel-count weights for data that the counting noise controls. The fit '
     'needs more time (about 30 s more with a GPU on a large volume).',
     'A comparison on the Averaging page. It is not a standard.'],
], [17, 28, 32, 23])}

<h2>2. Outlier rejection: which members go out</h2>
{_table(['Rule', 'What it does', 'What it removes', 'What it does not remove'], [
    ['<b>no rejection</b>',
     'All measured members go into the mean.',
     'Nothing.',
     'Spikes and zingers go into the mean and appear at all equivalent positions. The '
     'program never suggests this rule.'],
    ['<b>spread limit</b>',
     'A member goes out if it is farther from the median of the orbit than the threshold '
     'x the spread of the members. The spread is 1.4826 x MAD. It is at least '
     '&radic;max(median, 1), the counting noise of one pixel.',
     'Zingers, hot pixels and a bright arc that is in one member only.',
     'Extra intensity that is in one frame only and is smaller than the limit, for example '
     'the halo of a strong Bragg reflection. It can stay as a thin stripe. The limit cannot '
     'go below the counting noise of one pixel, but a voxel is the mean of many pixels.'],
    ['<b>noise limit</b>',
     'A member goes out if it is farther from the median than the threshold x its expected '
     'noise. The noise model gives the expected noise of each member from its pixel count. '
     'The member nearest the median always stays.',
     'Zingers, hot pixels and a bright arc in one member, and also extra intensity that is '
     'in one frame only, when it is small too.',
     'Contamination that is the same in many members, for example a complete powder ring. '
     'On a sharp peak its limit is wider (see the noise model below), so it can keep a '
     'member that the spread limit removes.'],
], [14, 32, 24, 30])}
<p>The spread limit and the noise limit need at least 3 measured members. With fewer members
the program does not reject.</p>

<h2>3. Threshold (sigma)</h2>
<p>The number in the name is the threshold. A higher value rejects less. The standard values
are 3 for the spread limit and 4 for the noise limit. In the windows you can set a value from
1 to 10.</p>

<h2>The standard options</h2>
{_table(['Data', 'Standard option', 'Reason'], [
    ['Raw frames (route A)', f'<i>{STD_RAW}</i>',
     'It removes intensity that is in one frame only. The tests used four raw-frame '
     'datasets (I15 and I19-2). In the diffuse region, the split-half difference was 0.5 to '
     '3.9 % lower with the noise limit than with the spread limit. Near the Bragg nodes it '
     f'was 10 to 18 % higher on the I15 datasets. If the fit of the noise model fails, the '
     f'window uses <i>{STD_RAW_FALLBACK}</i> and writes this in the log.'],
    ['Unwarp layers (route B)', f'<i>{STD_UNWARP}</i>',
     'The default of the earlier versions. The layers have no detector pixel counts. After a '
     'binning, the count of a voxel is only the number of its pixels with a measurement. '
     'Thus the pixel-count weights give almost no gain.'],
], [20, 30, 50])}

<h2>When to change the option</h2>
<ul>
<li><b>Thin stripes through the planes of a raw-frame volume.</b> They can come from extra
intensity in one frame. Use the noise limit. The spread limit does not remove this
intensity.</li>
<li><b>Powder rings or a background that changes in the scan.</b> The symmetry average does
not remove them fully. Use the <a href="topic:frame-tools">frame tools</a> (route A).</li>
<li><b>You are not sure.</b> Compare the options on your volume on the
<a href="topic:averaging-page">Averaging page</a>. It scores each option and suggests one.</li>
<li><b>Volumes that you subtract</b> (for example field on and field off) must use the same
option.</li>
</ul>

<h2>The noise model</h2>
<p>The noise limit and the noise weights use the variance of one member i at the median of
its orbit:</p>
<p><code>v<sub>i</sub> = gain &middot; max(median, 1) / N<sub>i</sub> + (rel &middot;
median)<sup>2</sup> + (kappa &middot; G)<sup>2</sup> + floor<sup>2</sup></code></p>
<p>N<sub>i</sub> is the pixel count of the member. G is the change of the intensity from
voxel to voxel. The term with G keeps the members on sharp peaks: these members sample the
peak at different positions in the voxel, so they scatter more. The program fits gain, rel,
kappa and floor to the volume. The Averaging page shows the fitted values after
<b>Measure</b>.</p>
'''

AVERAGING_PAGE = f'''
<h1>Compare the options on your data</h1>
<p>The <b>Averaging</b> page scores averaging options on your volume before you save it.
In route A, reconstruct the volume first: click <b>Reconstruct and save</b> once, and the
window keeps the volume in memory. In route B, select the unwarp folder first.</p>

<h2>The three steps</h2>
<ol>
<li><b>Measure.</b> The program fits the noise model and averages once with the reference
option <i>{REFERENCE}</i>. The tab <i>Diagnostics per |q| shell</i> shows the result.</li>
<li><b>Compare.</b> Tick the options in the list. Measure ticks the automatic set. Click
<b>Compare</b>. The tab <i>Comparison</i> shows a table, a suggestion and one slice. Click
a row of the table to see the slice of that option and its difference to the reference.</li>
<li><b>Apply.</b> Select an option (the suggestion is selected) and click <b>Apply</b>.
Then click the save button below it (<i>Symmetrise and save</i> or <i>Process all</i>). The main button of the first page also uses the
applied option. <b>Clear</b> removes the applied option.</li>
</ol>
<p><b>Cancel</b> stops Measure or Compare after the current step. The page keeps the earlier
results. A change of the Laue group, the bin factors or the volume removes the results.</p>

<h2>The Comparison table</h2>
{_table(['Column', 'Meaning', 'Good value'], [
    ['score', 'Each member is calculated from the other members of its orbit with the '
              'option. The score is the mean squared error divided by the expected '
              'variance, without the largest 1 %.', 'Lower is better.'],
    ['node', 'The sum near the Bragg nodes divided by the same sum without rejection. Each '
             'option with rejection removes a part of this sum, the reference option too.',
     'Not lower than the value of the reference row by more than 0.02.'],
    ['extra', 'The sum near the extra positions that you typed, divided by the sum of the '
              'reference option.', 'Within 2 % of 1.'],
    ['diffuse', 'The sum of the diffuse region divided by the sum of the reference option.',
     'Within 2 % of 1.'],
    ['lost', 'Voxels that have a value with the reference option and no value with this '
             'option.', '0'],
    ['no estimate', 'Sampled members for which the option gives no estimate.', '0'],
    ['time (s)', 'The time of the average with this option.', ''],
], [16, 58, 26])}
<p><b>The suggestion</b> is the option with the lowest score that passes all checks. An
option fails a check in these cases:</p>
<ul>
<li>Its extra sum or its diffuse sum differs from the reference by more than 2 %.</li>
<li>It loses voxels, or it gives no estimate for some members.</li>
<li>It removes more of the sum near the Bragg nodes than the reference does, by more than
2 % of the sum without rejection.</li>
</ul>
<p>If no option passes, the page suggests the safe option for the data. The page never
suggests an option without rejection.</p>
{_note(html.escape(av.LIMITS_TEXT), 'warn')}

<h2>The Diagnostics table</h2>
<p>The table has ten shells of the same width in |q| = 1/d.</p>
{_table(['Column', 'Meaning'], [
    ['chi2/nu', 'The scatter of the members against their variance. Near 1: the equivalent '
                'values agree within the noise. Much larger than 1: a scatter that the noise '
                'does not explain, for example a wrong Laue group or absorption. Smaller '
                'than 1: the variance is too large. For unwarp layers this number is only a '
                'relative measure.'],
    ['R_sym', 'The sum of |value - mean| divided by the sum of |value| over the members '
              'that remain. Lower is better.'],
    ['rejected %', 'The part of the members that the rejection removed.'],
    ['floor %', 'The part of the tested voxels where the counting noise of one pixel sets '
                'the spread limit.'],
    ['spread %', 'The part of the sampled orbits in which the largest pixel count of a '
                 'member is more than 10 x the smallest.'],
    ['node ratio', 'The sum near the Bragg nodes with the spread limit (3 sigma) divided by '
                   'the same sum without rejection.'],
], [18, 82])}

<h2>Speed</h2>
<p>With a GPU or with numba, Measure and Compare use the full volume. Without them, the page
uses a box at the centre of the volume, so that each option takes less than about one
minute.</p>
'''

FRAME_TOOLS = f'''
<h1>Frame tools</h1>
<p>Some contamination is in the beam for a part of the scan only. Examples are the powder
rings of a wire or of a sample holder, an added background, or a shadow. The symmetry average
does not remove
it fully, because each voxel gets it from different frames. The frame tools use the frames
again. They are on the <b>Frame tools</b> page of the raw-frame window. Reconstruct a volume
first.</p>
{_note('Validated scope: Laue group mmm and 0.2 degree frames (I15 and I19-2 data). Other '
       'Laue groups and frame widths work, but they are not validated.', 'warn')}

<h2>The tools</h2>
{_table(['Tool', 'What it does', 'Passes over the frames', 'New volume'], [
    ['<b>Frame map</b>',
     'Compares each frame with the other frames of the selected volume. Use it to see if '
     'some frames are different. The frames are not corrected in this pass.',
     '1 (the first use also needs the ring pass)', 'no'],
    ['<b>Ring mask</b>',
     'Finds powder rings in the frames and makes a volume without the ring pixels.',
     '2', '<i>ring mask</i>'],
    ['<b>Frame model</b>',
     'The ring mask, then 3 iterations of a model of the extra counts and of the scale of '
     'each frame and 5-degree sector.',
     '6 more', '<i>ring mask + frame model</i>'],
    ['<b>Split-half check</b>',
     'Makes the odd-frame and the even-frame halves of the selected volume and applies each '
     'candidate to each half. Click Measure on the Averaging page first.',
     '1', 'no'],
], [18, 46, 20, 16])}

<h2>The volume in use</h2>
<p>The list in the section <b>Volume in use</b> selects the volume that Measure, Compare and
the save steps use. It has the uncorrected volume and the volumes that Ring mask and Frame
model made. A change of the volume removes the results of Measure and Compare.</p>
<p>The file name of a corrected volume gets a tag. The tag is <code>_ringmask</code> for the
ring mask and <code>_ringmask_framemodel</code> for the frame model. It is
<code>_framemodel</code> if the ring mask found no ring.</p>

<h2>How to read the frame map</h2>
<p>{html.escape(av.FRAME_MAP_DEFINITION)}</p>
<p>The colour scale is logarithmic. White is 1. Red is more counts and blue is fewer counts
than in the other frames. The scale stops at 1/1.5 and 1.5, and a larger difference gets the
end colour. Grey cells have no prediction. On a long scan, one row of the image is a block of
frames. The row shows the value of the block that is farthest from 1. The plot on the right
is the residual against |q| for all frames, with a dashed line at each ring that the ring
mask found.</p>

<h2>The split-half difference</h2>
<p><code>{html.escape(av.SPLIT_HALF_DEFINITION)}</code></p>
<p>Lower is better. The split-half difference does not see contamination that is the same
in the two halves.</p>

<h2>Limits</h2>
<ul>
<li>The ring detection needs frames without the rings before or after the ring event in the
selected frames. If all selected frames have the rings, it finds no ring. The ring mask
depends on the selected frames, so check the number of ring frames after a change of the
frame count.</li>
<li>If Ring mask finds no ring, its volume is the same as the uncorrected volume, and Frame
model applies the frame model only.</li>
<li>A new reconstruction, a new folder, a new grid or a new frame count removes the results
of the frame tools. A new Laue group removes the frame-model volume and the frame map.</li>
<li>Memory: the split-half pass needs about 120 bytes for each voxel in the main memory
(about 55 bytes for a plain pass). On the GPU it needs 40 bytes for each voxel. Frame model
needed 6.0 GB more main memory in one measured run (1735 frames, grid 297 x 297 x 241).</li>
</ul>
'''

GRID = '''
<h1>Grid, voxel size, binning and memory</h1>

<h2>Range and voxel size (route A)</h2>
<p>The reconstruction puts the data on a regular grid in h, k and l. Set the smallest and
the largest value of each index in reciprocal-lattice units (r.l.u.). The ranges need not be
symmetric or equal. The <b>voxel size</b> is the step of the grid, the same on the three
axes. The default 0.025 r.l.u. resolves Bragg peaks and diffuse scattering.</p>
<p>The column <i>data reach</i> shows the largest |h|, |k| and |l| at the corners of the
detector. A range that is smaller than the reach cuts the outer data. A range that is larger
adds empty voxels.</p>

<h2>Memory</h2>
<p>On the GPU the reconstruction needs 20 bytes for each voxel. A grid of 481 x 481 x 481
voxels (-6 to 6 at 0.025) needs about 2.2 GB. The line <b>Grid size</b> shows the number of
voxels and this memory. It changes to the error colour when the grid needs more than 70 % of
the free GPU memory. To make the grid smaller, use narrower ranges or a larger voxel size. Half
the voxel size gives 8 x the voxels.</p>

<h2>Bin factors</h2>
<p>Binning puts neighbouring voxels together: fewer voxels, less noise, less resolution.</p>
<ul>
<li><b>Route A</b> (Advanced page): the bin factors are for the symmetrised volume. Set them
before you reconstruct. The grid then has hkl = 0 on a voxel centre after the binning, so
that the symmetry operations of all Laue groups are exact. The raw volume is not binned.</li>
<li><b>Route B</b> (Process page): the factor for h and k averages N x N pixels of each
layer, and the factor for l averages layers. 2 x 2 in the plane is the recommended
value.</li>
</ul>
'''

ADVANCED = '''
<h1>Advanced settings</h1>
<p>The defaults are the standard settings. <b>Reset to the standard settings</b> sets them
again.</p>

<h2>Raw-frame window</h2>
<table class="grid" cellspacing="0" cellpadding="6" width="100%">
<col width="30%"><col width="70%">
<tr><th align="left">Setting</th><th align="left">What it does and when to change it</th></tr>
<tr><td>Frames to use</td><td>The number of frames. 0 uses all frames. Use fewer frames for
a quick test.</td></tr>
<tr><td>Search subfolders for the .par file</td><td>Tick it if the
<code>*_cracker.par</code> is in a subfolder of the dataset folder.</td></tr>
<tr><td>Geometry from the CrysAlisPro peak table</td><td>Keep it ticked. The program fits
the position of the detector, the rotation axis and the orientation to the peak table. This
works with a wrong distance or beam centre in the frame header, and with powder rings.
Without it, the program uses the frame header, which works only for a clean dataset with a
correct header.</td></tr>
<tr><td>Use the GPU</td><td>Untick it to use the CPU. The result is the same. The CPU needs
about 1 s for each frame on a 481 x 481 x 481 grid.</td></tr>
<tr><td>Solid-angle and polarisation corrections</td><td>Each pixel is multiplied by the
geometric intensity correction. Keep it ticked.</td></tr>
<tr><td>Polarisation fraction</td><td>The part of the beam with horizontal polarisation.
About 0.95 for a synchrotron, 0.5 for a source without polarisation.</td></tr>
<tr><td>Hot-pixel limit</td><td>Pixels with more counts are ignored. 0 keeps all pixels. A
limit such as 1e6 removes zingers, but it also cuts saturated Bragg maxima on detectors that
count higher.</td></tr>
<tr><td>Bin factors</td><td>See <a href="topic:grid">Grid, voxel size, binning and
memory</a>.</td></tr>
<tr><td>Averaging and outlier rejection</td><td><i>Standard</i> or <i>Custom</i> (weights,
rejection rule and threshold). See <a href="topic:averaging">Averaging options</a>. An
option that you apply on the Averaging page replaces these settings.</td></tr>
</table>

<h2>Window for unwarp layers</h2>
<table class="grid" cellspacing="0" cellpadding="6" width="100%">
<col width="30%"><col width="70%">
<tr><th align="left">Setting</th><th align="left">What it does and when to change it</th></tr>
<tr><td>Averaging and outlier rejection</td><td>As above.</td></tr>
<tr><td>Mask kernel</td><td>The size (in pixels) that separates real low counts from regions
without a measurement (gaps, beam stop, edges). <i>auto</i> takes about 5 % of the distance
between two Bragg peaks. A larger value marks fewer pixels as not measured.</td></tr>
<tr><td>Read the .img files again</td><td>Without a tick, the window uses the raw volume
<code>&lt;sample&gt;_raw.h5</code> of an earlier run, if it agrees with the folder. Tick it
to read the <code>.img</code> files again, for example after CrysAlisPro wrote new
layers.</td></tr>
</table>
'''

VIEWER = '''
<h1>Look at a volume</h1>
<p>The viewer (<code>rspace3d-viewer</code>) opens a volume (<code>.h5</code>) of the two
routes, one CrysAlisPro layer (<code>.img</code>) and one raw frame (<code>.cbf</code>, shown
in pixels). <b>Open in viewer</b> in a processing window opens the last saved volume.</p>

<h2>The plane and the slice (volumes)</h2>
<p>The bar below the image selects the plane (hk, hl or kl) and the value of the third
index. Move the slider or type the value. <b>Integrate over &plusmn;</b> adds the slices in a
range around the value. Where a slice has no measurement, the sum uses the mean of the
other slices in its place.</p>

<h2>The colour scale</h2>
<p><b>Min</b> and <b>Max</b> are the ends of the colour scale. <b>Auto</b> sets them from
the slice that you see. <b>Log scale</b> shows weak diffuse scattering next to strong Bragg
peaks.</p>

<h2>The view</h2>
<p>The two range values are the half widths of the view in Miller-index units.
<b>Fit to data</b> shows all data of the plane. <b>Miller-index grid</b> draws a line at each
integer index. On a single layer, and on the plane of an unwarp volume that CrysAlisPro
calculated, the lines are tilted if the cell is not orthogonal. The other planes are shown
on axes of h, k and l, with straight lines.</p>

<h2>Tools</h2>
<ul>
<li><b>Line profile.</b> Switch it on and drag a line on the image. The panel on the right
shows the intensity along the line.</li>
<li><b>Export image.</b> Saves the image as a file or copies it to the clipboard.</li>
<li><b>3D view.</b> Opens the volume in 3D in its own window: surfaces of equal intensity
or a projection. See <a href="topic:view3d">Look at a volume in 3D</a>.</li>
<li><b>File information.</b> Shows the size, the ranges, the unit cell and the processing
data of the volume.</li>
</ul>
<p>The status bar shows the indices and the intensity below the cursor.</p>
'''

VIEW3D = f'''
<h1>Look at a volume in 3D</h1>
<p><b>3D view</b> in the viewer opens the volume in its own window. The window draws on the
graphics card. A change of a level, of the box or of the direction shows at once, also for
a large volume.</p>

<h2>What you see</h2>
<p>The box has the axes h, k and l. It is drawn in reciprocal space with the unit cell of
the volume, so the box of a monoclinic or triclinic cell does not have right angles. The
lines on the far faces of the box are at the whole Miller indices. The small arrows at the
lower left show the directions of a*, b* and c*.</p>

<h2>Surfaces of equal intensity</h2>
<p>A surface is where the intensity has the value of a level. Each point inside the surface
is stronger than the level. A sharp Bragg peak gives a small closed surface. A diffuse rod
gives a tube, and a diffuse plane gives a sheet.</p>
<ul>
<li><b>The histogram</b> shows how many voxels have each intensity, on a logarithmic axis.
Each level is a line on it. Drag the line, or type the intensity in the field below it.
Hold Shift during the drag for small steps.</li>
<li><b>The first level</b> is set so that 1 % of the voxels with a positive intensity are
above it. These voxels are the Bragg peaks and the strongest diffuse scattering. <b>Reset
levels</b> removes the other levels and sets this level again.</li>
<li><b>Add level</b> adds a second and a third surface, each with its colour and opacity. A
good use is one surface with an opacity of 100 for the strong features, and one surface with
a low opacity at a lower level for the weak scattering around them. A level near the large
peak at the left of the histogram is in the noise and shows a rough cloud.</li>
<li><b>Grey parts</b> of a surface are where the measured data ends: a gap of the detector,
the beam stop or the limit of the measurement. The intensity does not decrease there. There
is no data on the other side.</li>
<li><b>Flat faces</b> are where the box cuts a region that is above the level.</li>
</ul>
{_note('The 3D view does not change the data. It finds a surface between the voxel centres '
       'on a logarithmic scale and applies no smoothing.')}

<h2>Projection of the maximum</h2>
<p>Each point of the image shows the largest intensity along the line of sight, in the
colours of the colour map. You do not select a level, so this is a good first look at weak
features in all of the volume. A strong feature hides the weak features in front of it and
behind it. The two lines on the histogram are the ends of the colour scale (<b>Min</b> and
<b>Max</b>). On a light background the colour map is reversed, so that weak intensity is
light.</p>

<h2>The box</h2>
<p>The three ranges cut the volume along h, k and l. Use them to look inside. For example,
set l from 0 to its largest value: the cut face is then the plane l = 0. <b>Show all</b>
shows the complete volume again. If you open the window from the viewer, the two ranges of
the 2D view are the start values.</p>

<h2>The view</h2>
{_table(['Mouse', 'Effect'], [
    ['Drag', 'Turn the box. The direction at right angles to the hk plane stays upright.'],
    ['Shift + drag, or drag with the right button', 'Move the box.'],
    ['Wheel', 'Zoom. The point below the mouse stays in its place.'],
], [42, 58])}
<p><b>3D</b> sets the start direction again. <b>hk</b>, <b>hl</b> and <b>kl</b> look straight
at that plane, with the first index to the right and the second index up, as in the 2D
view. Without <b>Perspective</b>, equal lengths are equal on the screen at each depth. This
is the usual way to draw a lattice.</p>
<p>The line below the image shows h, k, l and the intensity of the surface point below the
mouse. In the projection it shows the strongest point along the line of sight.</p>

<h2>Image</h2>
<p><b>Save image...</b> writes a PNG file with the labels. The file has two image pixels for
each screen pixel, and not more than 4096 pixels along its longer side. <b>Copy image</b>
puts the same image on the clipboard.</p>

<h2>Speed and size</h2>
<ul>
<li>The view needs graphics with OpenGL 4.1. It needs no extra Python package.</li>
<li>If a frame takes longer than 1/30 s, the image has fewer pixels while you turn it and
all pixels when you stop.</li>
<li>A volume with more than about 500 million voxels, or with an axis of more than 2048
voxels, is shown as the mean of blocks of voxels. The panel then says "Shown with 1 of
each ...".</li>
<li>From Python, <code>rspace3d.volume_isosurface.plot_isosurface(volume)</code> opens the
same window. The older methods <code>method='pyvista'</code> and
<code>method='plotly'</code> are still there. They are slow for a large volume. A call with
<code>save_html</code> and no method uses Plotly, as before.</li>
</ul>
'''

FILES = f'''
<h1>Output files</h1>
{_table(['File', 'Content'], [
    ['<code>&lt;folder&gt;_recon_sym_&lt;laue&gt;.h5</code> (route A)<br>'
     '<code>&lt;sample&gt;_sym_&lt;laue&gt;.h5</code> (route B)',
     'The volume after the symmetry average and the outlier rejection. This is the file for '
     'the viewer and for your analysis. The Laue group is in the name, for example '
     '<code>mbar3m</code> for m-3m.'],
    ['<code>&lt;folder&gt;_recon_raw.h5</code> (route A)<br>'
     '<code>&lt;sample&gt;_raw.h5</code> (route B)',
     'The volume before the average. With it you can average again later with another Laue '
     'group or another option, without a new reconstruction. Keep it for each analysis that '
     'looks for a break of the symmetry.'],
], [42, 58])}
<p>In route A the files go into the output folder of step 5 (default: the dataset folder).
In route B they go into the unwarp folder.</p>

<h2>In the files</h2>
<ul>
<li><code>/data</code>: the intensity, in the order (h, k, l). A voxel without a measurement
is <code>NaN</code>. A measured voxel with zero counts is 0.</li>
<li><code>/H</code>, <code>/K</code>, <code>/L</code>: the Miller indices of the three
axes.</li>
<li><code>/uncertainty</code>: the standard deviation of each voxel value. The raw file of
route B does not have it.</li>
<li><code>/counts</code>: the number of pixels in each voxel, if the volume has pixel
counts.</li>
<li><code>/UB</code> and the attributes: the UB matrix, the unit cell, the wavelength, the
Laue group, the averaging option and its threshold, and the number of rejected members.</li>
</ul>
<p>The files are HDF5. You can read them in Python with <code>h5py</code> or with
<code>rspace3d.load_volume_h5</code>, and in MATLAB with <code>h5read(file, '/data')</code>.</p>
'''

GLOSSARY = f'''
<h1>Glossary</h1>
{_table(['Term', 'Meaning'], [
    ['Bragg node', 'A position with integer h, k and l, where a Bragg peak is.'],
    ['Diffuse region', 'The voxels that are not within 0.1 (in hkl) of a Bragg node or of '
                       'an extra position that you typed.'],
    ['Laue group', 'The symmetry of the diffraction pattern. It gives the equivalent '
                   'positions that the average uses.'],
    ['MAD', 'Median absolute deviation: the median of the distances of the members from '
            'their median. 1.4826 x MAD estimates the standard deviation and is not '
            'sensitive to a few outliers.'],
    ['Member', 'One measured position of an orbit.'],
    ['Noise model', 'A formula for the variance of a member, with four parameters that the '
                    'program fits to the volume. See <a href="topic:averaging">Averaging '
                    'options</a>.'],
    ['Orbit', 'A voxel and all its symmetry-equivalent positions.'],
    ['Pixel count', 'The number of detector pixels that went into a voxel.'],
    ['Poisson floor', 'The lowest value of the spread in the spread limit: '
                      '&radic;max(median, 1), the counting noise of one pixel.'],
    ['r.l.u.', 'Reciprocal-lattice unit: the unit of h, k and l.'],
    ['Raw volume', 'The volume before the symmetry average.'],
    ['Reference option', f'<i>{REFERENCE}</i>. The checks of the Averaging page compare '
                         'each option with it.'],
    ['Sigma', 'The threshold of the outlier rejection. A higher value rejects less.'],
    ['Split-half difference', 'A measure of the noise from two volumes: the odd frames and '
                              'the even frames. See <a href="topic:frame-tools">Frame '
                              'tools</a>.'],
    ['Unwarp layer', 'A plane of reciprocal space that CrysAlisPro calculates with '
                     '<code>dc unwarp</code>.'],
    ['Voxel', 'One cell of the 3D grid.'],
    ['Zinger', 'A very high count in one or a few pixels of one frame that is not from the '
               'sample.'],
], [22, 78])}
'''

EXPLORER_INTRO = (
    'One voxel and the members of its orbit. Each dot is one member: the value that one '
    'symmetry-equivalent position measured. Select a case and an option, and see which '
    'members go out and what value the voxel gets. The calculation is the calculation of '
    'the program.')

TOPICS: list[tuple[str, str, str]] = [
    ('start', 'Start here', START),
    ('raw-frames', 'Reconstruct from raw frames', RAW_FRAMES),
    ('unwarp', 'Volume from CrysAlisPro layers', UNWARP),
    ('symmetry', 'Symmetry average and Laue group', SYMMETRY),
    ('averaging', 'Averaging options', AVERAGING),
    (EXPLORER, 'Try the options on one voxel', ''),
    ('averaging-page', 'Compare the options on your data', AVERAGING_PAGE),
    ('frame-tools', 'Frame tools', FRAME_TOOLS),
    ('grid', 'Grid, voxel size, binning and memory', GRID),
    ('advanced', 'Advanced settings', ADVANCED),
    ('viewer', 'Look at a volume', VIEWER),
    ('view3d', 'Look at a volume in 3D', VIEW3D),
    ('files', 'Output files', FILES),
    ('glossary', 'Glossary', GLOSSARY),
]


def topic_keys() -> list[str]:
    return [key for key, _, _ in TOPICS]


def topic_title(key: str) -> str:
    return next(title for k, title, _ in TOPICS if k == key)


def topic_html(key: str) -> str:
    """The HTML of a topic. An unknown key gives the first topic."""
    for k, _, text in TOPICS:
        if k == key:
            return text
    return TOPICS[0][2]
