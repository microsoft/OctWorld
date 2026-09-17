const galleryGroups = [
  {
    gridId: "dl3dv-grid",
    batches: [
      { directory: "dl3dv_case", cases: [2, 9] }
    ]
  },
  {
    gridId: "real-grid",
    batches: [
      { directory: "real_style", cases: [14, 19] }
    ]
  },
  {
    gridId: "synthetic-grid",
    batches: [
      { directory: "real_style", cases: [7] },
      { directory: "synthetic_style", cases: [30, 32, 45, 50, 53, 56, 58, 61, 63, 65, 70, 73, 78] }
    ]
  }
];

const VIDEO_VERSION = "crf24-1";
const heroVideos = [...document.querySelectorAll(".intro-background video")];
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const heroSources = galleryGroups.flatMap(({ batches }) => batches).filter(({ directory }) => (
  directory !== "dl3dv_case"
)).flatMap(({ directory, cases }) => (
  cases.map((caseId) => `assets/${directory}/case_${caseId}/output_rgb_30fps.mp4`)
));
let heroQueue = [];
let activeHeroVideo = 0;
let heroTimer = null;
let heroIsVisible = true;
let heroTransitioning = false;

function posterForVideo(source) {
  const posterName = source
    .replace(/^assets\//, "")
    .replace(/\//g, "--")
    .replace(/\.mp4$/, ".webp");
  return `assets/posters/${posterName}`;
}

function shuffle(items) {
  const shuffled = [...items];
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const randomIndex = Math.floor(Math.random() * (index + 1));
    [shuffled[index], shuffled[randomIndex]] = [shuffled[randomIndex], shuffled[index]];
  }
  return shuffled;
}

function nextHeroSource() {
  if (heroQueue.length === 0) heroQueue = shuffle(heroSources);
  return heroQueue.pop();
}

function loadHeroSource(video, source) {
  return new Promise((resolve) => {
    let timeoutId;
    const finish = (ready) => {
      window.clearTimeout(timeoutId);
      video.removeEventListener("loadeddata", onLoaded);
      video.removeEventListener("error", onError);
      resolve(ready);
    };
    const onLoaded = () => finish(true);
    const onError = () => finish(false);

    video.addEventListener("loadeddata", onLoaded, { once: true });
    video.addEventListener("error", onError, { once: true });
    video.poster = posterForVideo(source);
    video.src = `${source}?v=${VIDEO_VERSION}`;
    video.load();
    timeoutId = window.setTimeout(() => finish(false), 8000);
  });
}

function scheduleHeroTransition() {
  window.clearTimeout(heroTimer);
  if (heroIsVisible && !document.hidden && !reduceMotion.matches) {
    heroTimer = window.setTimeout(() => transitionHeroVideo(), 22000);
  }
}

async function transitionHeroVideo(initial = false) {
  if (heroTransitioning) return;
  heroTransitioning = true;
  const nextIndex = initial ? 0 : 1 - activeHeroVideo;
  const nextVideo = heroVideos[nextIndex];
  let ready = false;

  for (let attempt = 0; attempt < 3 && !ready; attempt += 1) {
    ready = await loadHeroSource(nextVideo, nextHeroSource());
  }
  if (!ready) {
    heroTransitioning = false;
    scheduleHeroTransition();
    return;
  }

  nextVideo.currentTime = 0;
  if (heroIsVisible && !document.hidden && !reduceMotion.matches) {
    await nextVideo.play().catch(() => undefined);
  }
  nextVideo.classList.add("is-active");

  if (!initial) {
    heroVideos[activeHeroVideo].classList.remove("is-active");
    heroVideos[activeHeroVideo].pause();
  }
  activeHeroVideo = nextIndex;
  heroTransitioning = false;
  scheduleHeroTransition();
}

const heroObserver = new IntersectionObserver(([entry]) => {
  heroIsVisible = entry.isIntersecting;
  window.clearTimeout(heroTimer);
  if (!heroIsVisible) {
    heroVideos.forEach((video) => video.pause());
    return;
  }

  if (!document.hidden && !reduceMotion.matches) {
    heroVideos[activeHeroVideo].play().catch(() => undefined);
  }
  scheduleHeroTransition();
}, { threshold: 0.05 });

heroObserver.observe(document.querySelector(".intro"));
transitionHeroVideo(true);

const methodVideos = [...document.querySelectorAll(".method-media video")];
let visibleMethodVideo = null;
const methodMediaObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    const video = entry.target;
    const shouldPlay = !document.hidden && entry.isIntersecting && entry.intersectionRatio >= 0.45;
    if (!shouldPlay) {
      video.pause();
      if (visibleMethodVideo === video) visibleMethodVideo = null;
      return;
    }

    visibleMethodVideo = video;
    methodVideos.forEach((otherVideo) => {
      if (otherVideo !== video) otherVideo.pause();
    });
    if (video.preload === "none") {
      video.preload = "metadata";
      video.load();
    }
    if (!reduceMotion.matches) video.play().catch(() => undefined);
  });
}, { threshold: [0, 0.45] });

methodVideos.forEach((video) => methodMediaObserver.observe(video));

const sceneTemplate = document.querySelector("#scene-template");
const syncGroups = [];
const canHover = window.matchMedia("(hover: hover)");
const galleryLoadQueue = [];
const loadingGalleryPairs = new Set();
const galleryLoadConcurrency = 2;
let activePair = null;

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    window.clearTimeout(heroTimer);
    heroVideos.forEach((video) => video.pause());
    methodVideos.forEach((video) => video.pause());
    if (activePair) pausePair(activePair);
    return;
  }

  if (heroIsVisible && !reduceMotion.matches) {
    heroVideos[activeHeroVideo].play().catch(() => undefined);
    scheduleHeroTransition();
  } else if (visibleMethodVideo && !reduceMotion.matches) {
    visibleMethodVideo.play().catch(() => undefined);
  }
});

function createSceneCell({ source, caseId, position }) {
  const cell = sceneTemplate.content.firstElementChild.cloneNode(true);
  const video = cell.querySelector("video");
  const fallback = cell.querySelector(".video-fallback");
  const toggle = cell.querySelector(".play-toggle");

  cell.dataset.position = position;
  video.dataset.src = `${source}?v=${VIDEO_VERSION}`;
  video.dataset.poster = posterForVideo(source);
  video.setAttribute(
    "aria-label",
    `${position === "generation" ? "Generated scene" : "Octree map visualization"}, case ${caseId}`
  );
  video.addEventListener("error", () => {
    if (video.dataset.preloadCancelled === "true" || video.dataset.preloading === "true") return;
    fallback.hidden = false;
    toggle.hidden = true;
  });

  return { cell, video, toggle };
}

function buildPair(caseId, directory) {
  const basePath = `assets/${directory}/case_${caseId}`;
  const generation = createSceneCell({
    source: `${basePath}/output_rgb_30fps.mp4`,
    caseId,
    position: "generation"
  });
  const map = createSceneCell({
    source: `${basePath}/octmap_integration_30fps.mp4`,
    caseId,
    position: "map"
  });
  const pair = {
    master: generation.video,
    follower: map.video,
    cells: [generation.cell, map.cell],
    playRequest: 0,
    isVisible: false,
    mapVisible: false
  };

  syncGroups.push(pair);
  attachPairSyncListeners(pair);
  [generation.toggle, map.toggle].forEach((button) => {
    button.addEventListener("click", () => togglePair(pair));
  });

  const pairCard = document.createElement("article");
  pairCard.className = "scene-pair is-map-hidden";

  const videoBadge = document.createElement("span");
  videoBadge.className = "video-mode-badge";
  videoBadge.textContent = "Generated Video";

  const mapExpand = document.createElement("button");
  mapExpand.className = "map-expand";
  mapExpand.type = "button";
  mapExpand.setAttribute("aria-label", "Expand 3D map");
  mapExpand.setAttribute("aria-pressed", "false");
  mapExpand.title = "Expand 3D map";
  mapExpand.innerHTML = '<svg class="expand-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 3H3v5M16 3h5v5M21 16v5h-5M3 16v5h5"></path></svg><svg class="collapse-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 3v5H3M16 3v5h5M21 16h-5v5M3 16h5v5"></path></svg>';
  mapExpand.addEventListener("click", () => {
    const expanded = pairCard.classList.toggle("is-map-expanded");
    const label = `${expanded ? "Restore" : "Expand"} 3D map`;
    mapExpand.setAttribute("aria-label", label);
    mapExpand.setAttribute("aria-pressed", String(expanded));
    mapExpand.title = label;
  });

  const mapToggle = document.createElement("button");
  mapToggle.className = "map-toggle";
  mapToggle.type = "button";
  mapToggle.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12Z"></path><circle cx="12" cy="12" r="2.5"></circle></svg><span>Show 3D map</span>';
  mapToggle.setAttribute("aria-pressed", "false");
  mapToggle.addEventListener("click", () => {
    const hidden = pairCard.classList.toggle("is-map-hidden");
    pair.mapVisible = !hidden;
    if (hidden) {
      pairCard.classList.remove("is-map-expanded");
      mapExpand.setAttribute("aria-label", "Expand 3D map");
      mapExpand.setAttribute("aria-pressed", "false");
      mapExpand.title = "Expand 3D map";
      pair.follower.pause();
    } else {
      prepareMapPlayback(pair);
    }
    mapToggle.setAttribute("aria-pressed", String(!hidden));
    mapToggle.querySelector("span").textContent = `${hidden ? "Show" : "Hide"} 3D map`;
  });

  map.cell.querySelector(".video-frame").append(mapExpand);
  generation.cell.querySelector(".video-frame").append(videoBadge, map.cell, mapToggle);
  pairCard.append(generation.cell);
  pairCard.addEventListener("mouseenter", () => {
    if (canHover.matches) {
      prioritizePairPlayback(pair);
      playPair(pair);
    }
  });
  pairCard.addEventListener("mouseleave", () => {
    if (canHover.matches) pausePair(pair, true);
  });
  pairCard.pair = pair;
  return pairCard;
}

const mediaObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    const pair = entry.target.pair;
    pair.isVisible = entry.isIntersecting;
    if (entry.isIntersecting) {
      [pair.master, pair.follower].forEach(loadPoster);
      const group = entry.target.closest(".gallery-group");
      if (group?.dataset.preloadActive === "true") requestPairPreload(pair);
    } else {
      cancelStalePairPreload(pair);
      if (activePair === pair) pausePair(pair);
    }
  });
}, { rootMargin: "400px 0px", threshold: 0.01 });

function itemsForGroup(batches) {
  return batches.flatMap(({ directory, cases }) => (
    cases.map((caseId) => ({ caseId, directory }))
  ));
}

function setupGalleryGroup({ gridId, batches }) {
  const grid = document.querySelector(`#${gridId}`);
  const group = grid.closest(".gallery-group");
  const items = itemsForGroup(batches);

  function preloadCurrentGroup() {
    group.dataset.preloadActive = "true";
    [...grid.children].forEach((pairCard) => requestPairPreload(pairCard.pair));
  }

  group.addEventListener("mouseenter", preloadCurrentGroup);
  group.addEventListener("focusin", preloadCurrentGroup);
  items.forEach(({ caseId, directory }) => {
    const pairCard = buildPair(caseId, directory);
    grid.append(pairCard);
    mediaObserver.observe(pairCard);
  });
}

galleryGroups.forEach(setupGalleryGroup);

function setPairState(pair, playing) {
  pair.cells.forEach((cell) => cell.classList.toggle("is-playing", playing));
  pair.cells.forEach((cell) => {
    const button = cell.querySelector(":scope > .video-frame > .play-toggle");
    if (button) {
      button.setAttribute("aria-label", `${playing ? "Pause" : "Play"} paired scene`);
      button.title = `${playing ? "Pause" : "Play"} paired scene`;
    }
  });
}

function loadVideo(video) {
  if (video.error) {
    video.removeAttribute("src");
    video.load();
    const frame = video.closest(".video-frame");
    const fallback = frame?.querySelector(":scope > .video-fallback");
    const toggle = frame?.querySelector(":scope > .play-toggle");
    if (fallback) fallback.hidden = true;
    if (toggle) toggle.hidden = false;
  }
  if (!video.hasAttribute("src")) {
    delete video.dataset.preloadCancelled;
    video.src = video.dataset.src;
    video.load();
  }
}

function requestPairPreload(pair) {
  if (pair.master.hasAttribute("src") || pair.preloadQueued) return;
  pair.preloadQueued = true;
  galleryLoadQueue.unshift(pair);
  processGalleryLoadQueue();
}

function cancelStalePairPreload(pair) {
  if (
    !loadingGalleryPairs.has(pair)
    || activePair === pair
    || pair.master.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
  ) return;

  pair.master.dataset.preloadCancelled = "true";
  pair.master.pause();
  pair.master.removeAttribute("src");
  pair.master.preload = "none";
  pair.master.load();
}

function prioritizePairPlayback(pair) {
  if (loadingGalleryPairs.has(pair) || loadingGalleryPairs.size < galleryLoadConcurrency) return;
  const backgroundPair = [...loadingGalleryPairs].find((loadingPair) => loadingPair !== pair);
  if (backgroundPair) cancelStalePairPreload(backgroundPair);
}

function processGalleryLoadQueue() {
  while (loadingGalleryPairs.size < galleryLoadConcurrency && galleryLoadQueue.length) {
    const pair = galleryLoadQueue.shift();
    pair.preloadQueued = false;
    if (!pair.isVisible || pair.master.hasAttribute("src")) continue;

    loadingGalleryPairs.add(pair);
    pair.master.preload = "auto";
    pair.master.dataset.preloading = "true";
    waitForLoadedData(pair.master, 5000).finally(() => {
      delete pair.master.dataset.preloading;
      loadingGalleryPairs.delete(pair);
      processGalleryLoadQueue();
    });
  }
}

function loadPoster(video) {
  if (!video.hasAttribute("poster")) video.poster = video.dataset.poster;
}

function waitForLoadedData(video, timeout) {
  loadVideo(video);
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) return Promise.resolve(true);
  if (video.error) return Promise.resolve(false);

  return new Promise((resolve) => {
    let timeoutId;
    const finish = (ready) => {
      window.clearTimeout(timeoutId);
      video.removeEventListener("loadeddata", onLoaded);
      video.removeEventListener("error", onError);
      video.removeEventListener("abort", onAbort);
      resolve(ready);
    };
    const onLoaded = () => finish(true);
    const onError = () => finish(false);
    const onAbort = () => finish(false);
    video.addEventListener("loadeddata", onLoaded, { once: true });
    video.addEventListener("error", onError, { once: true });
    video.addEventListener("abort", onAbort, { once: true });
    timeoutId = window.setTimeout(() => finish(false), timeout);
  });
}

function waitForMetadata(video, timeout = 12000) {
  loadVideo(video);
  if (video.readyState >= HTMLMediaElement.HAVE_METADATA && Number.isFinite(video.duration)) {
    return Promise.resolve(true);
  }
  if (video.error) return Promise.resolve(false);

  return new Promise((resolve) => {
    let timeoutId;
    const finish = (ready) => {
      window.clearTimeout(timeoutId);
      video.removeEventListener("loadedmetadata", onLoaded);
      video.removeEventListener("error", onError);
      video.removeEventListener("abort", onAbort);
      resolve(ready);
    };
    const onLoaded = () => finish(true);
    const onError = () => finish(false);
    const onAbort = () => finish(false);
    video.addEventListener("loadedmetadata", onLoaded, { once: true });
    video.addEventListener("error", onError, { once: true });
    video.addEventListener("abort", onAbort, { once: true });
    timeoutId = window.setTimeout(() => finish(false), timeout);
  });
}

async function playPair(pair) {
  if (activePair && activePair !== pair) pausePair(activePair, true);
  activePair = pair;
  const requestId = ++pair.playRequest;
  const playbackVideos = pair.mapVisible ? [pair.master, pair.follower] : [pair.master];
  const ready = await Promise.all(playbackVideos.map(waitForMetadata));
  if (requestId !== pair.playRequest || activePair !== pair || ready.includes(false)) return;

  playbackVideos.forEach((video) => video.pause());
  pair.master.currentTime = 0;
  if (pair.mapVisible) {
    pair.follower.currentTime = 0;
    syncFollower(pair, true);
  }
  await Promise.all(
    playbackVideos.map((video) => video.play().catch(() => undefined))
  );
  if (requestId !== pair.playRequest || activePair !== pair) {
    playbackVideos.forEach((video) => video.pause());
    return;
  }
  setPairState(pair, !pair.master.paused);
}

async function prepareMapPlayback(pair) {
  pair.follower.preload = "metadata";
  if (!await waitForMetadata(pair.follower) || !pair.mapVisible) return;
  syncFollower(pair, true);
  if (activePair === pair && !pair.master.paused) {
    await pair.follower.play().catch(() => undefined);
  }
}

function pausePair(pair, reset = false) {
  pair.playRequest += 1;
  pair.master.pause();
  pair.follower.pause();
  if (reset && pair.master.readyState >= HTMLMediaElement.HAVE_METADATA) pair.master.currentTime = 0;
  if (reset && pair.follower.readyState >= HTMLMediaElement.HAVE_METADATA) pair.follower.currentTime = 0;
  if (activePair === pair) activePair = null;
  setPairState(pair, false);
}

function togglePair(pair) {
  if (pair.master.paused) playPair(pair);
  else pausePair(pair, true);
}

function syncFollower(pair, force = false) {
  if (!pair.master.duration || !pair.follower.duration) return;
  pair.follower.playbackRate = pair.follower.duration / pair.master.duration;
  const targetTime = (pair.master.currentTime / pair.master.duration) * pair.follower.duration;
  if (force || Math.abs(pair.follower.currentTime - targetTime) > 0.35) {
    pair.follower.currentTime = targetTime;
  }
}

function attachPairSyncListeners(pair) {
  pair.master.addEventListener("timeupdate", () => syncFollower(pair));
  pair.master.addEventListener("pause", () => {
    pair.follower.pause();
    setPairState(pair, false);
  });
  pair.master.addEventListener("play", () => setPairState(pair, true));
  pair.master.addEventListener("seeked", () => syncFollower(pair));
}

document.querySelector(".copy-button").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  const citation = document.querySelector(".citation-block code").textContent;
  await navigator.clipboard.writeText(citation);
  button.title = "Copied";
  button.setAttribute("aria-label", "BibTeX copied");
  window.setTimeout(() => {
    button.title = "Copy BibTeX";
    button.setAttribute("aria-label", "Copy BibTeX");
  }, 1600);
});