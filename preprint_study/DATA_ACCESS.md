# Provider inputs and redistribution

Checked 31 October 2026 against the providers' published terms:

- [Yahoo Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html)
  restrict commercial exploitation and uses that create material substitutes for
  Yahoo or its data providers. The [Developer API terms](https://legal.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.html)
  do not establish a blanket right to redistribute the Finance chart history.
  The chart endpoint used here is unofficial and may change or deny access.
- [Cboe website terms, section 2](https://www.cboe.com/terms) permit a personal,
  noncommercial copy and restrict other copying, publishing, and distribution
  without consent, subject to the stated exceptions. A publicly downloadable
  history file is not a general redistribution licence.

The release therefore contains **no provider observations, adjusted price
files, raw API responses, or date-by-date market/forecast/portfolio series**.
It provides code, source URLs, dated hashes, trained model parameters, fitted
structural parameters, and aggregate research summaries and figures. Neither
this repository nor its archive grants rights to third-party market data.

## Obtain local inputs

Review the terms and obtain appropriate access rights. If appropriate for your
use, run:

```sh
python download.py --acknowledge-terms
python verify_snapshot.py
```

The acknowledgment is an explicit step for the person running the script. It
does not establish that a particular use is authorized. The script does not
bypass access restrictions or retry blocked requests through alternative hosts.
Alternatively, supply lawfully obtained local files in `data/`:

- `SPY.csv`, `IWM.csv`, `TLT.csv`, `GLD.csv`: `Date` index with adjusted `Open`,
  `High`, `Low`, `Close`, and `Volume`; daily history from 2012 through 2025.
- `VIX.csv`: Cboe-format `DATE` and `CLOSE` columns. Extra historical/future
  dates are allowed; predictors align causally to the ETF calendar.

`data/snapshot_manifest.json` records the original file hashes and source URLs.
Its five canonical hashes cover the dated Open/Close input panels and aligned
VIX values actually used. Appending later VIX observations or changing an API
response's retrieval metadata does not change those canonical inputs.

Provider histories and adjustment factors may be revised. Exact historical
numerical reproduction is conditional on matching canonical input hashes.
`verify_snapshot.py` fails explicitly when they differ; it does not silently
label a current-data rerun an exact replication. The downloader writes a
separate, ignored `data/local_manifest.json` and preserves the published
snapshot manifest. A fresh current-data rerun can still evaluate the procedure,
but its results should be identified as a new dataset version.

The public results are research summaries, not a substitute market-data feed.
Local files and date-by-date experimental caches are ignored by Git and excluded
by the release packager. This policy does not assert a blanket permission for
all possible uses of either provider's data.
