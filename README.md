# OctWorld project website

This branch contains the static OctWorld project page. Preview it locally with
`python serve.py --bind 127.0.0.1 --port 8080`.

## Browser storage and network inventory

Reviewed on 2026-10-20 against `index.html`, `styles.css`, and `script.js`.

| Feature | Current implementation |
| --- | --- |
| Cookies set by site code | None, either essential or non-essential. |
| Browser storage | No localStorage, sessionStorage, IndexedDB, or service worker use. |
| Analytics and advertising | No analytics SDKs, advertising scripts, or tracking pixels. |
| Fonts, images, videos, and scripts | Served from the site's own origin; no third-party video embeds or remote asset dependencies. |
| Visitor input | No forms, accounts, registration, or data-collecting pop-ups. The citation button writes BibTeX to the clipboard only when clicked. |
| External destinations | Microsoft policy pages, the Microsoft homepage, and the research paper open on link activation. Contact Us opens the visitor's email client. |

Local Chromium checks found no cookies or browser storage entries and no
third-party network requests while loading the page. This inventory covers the
site code; it does not establish the behavior of hosting infrastructure, access
controls, or authentication redirects. Recheck the deployed site, including its
response headers and browser storage, when hosting or access settings change.
If non-essential cookies or analytics are introduced, update the inventory and
implement consent before enabling them.

## Footer assets and maintenance

`assets/microsoft-logo.png` is the unmodified full-color Microsoft logotype from
[Microsoft's official asset host](https://uhf.microsoft.com/images/microsoft/RE1Mu3b.png),
served locally and linked to the Microsoft homepage.

The header icon and favicon use `assets/microsoft-symbol.png`, the square
four-color symbol cropped from that official logotype, with its colors and
layout preserved.

The Terms of Use fwlink (`206977`) was verified to resolve to Microsoft's
[Terms of Use](https://www.microsoft.com/en-us/legal/terms-of-use).
Keep all footer links available on small screens and without JavaScript, and
update the static Microsoft copyright year when appropriate. The project owner
must ensure the Contact Us mailbox is monitored.
