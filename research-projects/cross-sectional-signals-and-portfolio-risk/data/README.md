# Data provenance

Input is the existing repository snapshot at `../capital-shortfall-under-geopolitical-stress/data/48_Industry_Portfolios_Daily.csv` (path relative to project root). This avoids redistributing a duplicate 13MB file. Original source: Kenneth French Data Library, https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_48_ind_port.html . Copyright Eugene F. Fama and Kenneth R. French.

This CSV contains two return blocks. The second is explicitly labelled Average Equal Weighted Returns -- Daily. The first is used as the value-weighted-layout block, consistent with French's file ordering; its heading and original download metadata were removed in the inherited CSV. Consequently exact source vintage and independent verification of the original value-weighted heading are unavailable. The study records its SHA-256 and never averages duplicate dates across these blocks. A fresh provider download failed during preparation; this is not labelled current data.

Only 1990–2024 observations enter the analysis. Input returns are percentages; the loader converts them to decimal, checks missing-value sentinels, uniqueness and finiteness, and fails rather than silently filling missing returns.

Industry constituents are reconstructed historically by the provider, rather than selected from today's survivors, but the snapshot can contain retrospective revisions and must not be called point-in-time data. Aggregate returns are research proxies, not exchange-traded instruments.
