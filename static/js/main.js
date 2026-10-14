/**
 * LumiPix — main.js
 * Controls upload, filter selection, parameter binding,
 * API calls, comparison slider, auth modal, and metrics display.
 */

document.addEventListener("DOMContentLoaded", () => {

    /* ── State ─────────────────────────────────────────────────────── */
    const state = {
        currentUser: null,
        uploadedFile: null,
        selectedSampleId: null,
        activeFilter: "adaptive",
        downloadUrl: null,
        downloadFilename: "lumipix_restored.png",
        splitPct: 50,
        isDragging: false,
        originalDataUrl: null,
        restoredDataUrl: null,
    };

    /* ── DOM refs ──────────────────────────────────────────────────── */
    const $ = (id) => document.getElementById(id);
    const dropzone          = $("dropzone");
    const fileInput         = $("fileInput");
    const dropzonePreview   = $("dropzonePreview");
    const dropzoneDefault   = $("dropzoneDefault");
    const sourcePreviewImg  = $("sourcePreviewImg");
    const previewFileName   = $("previewFileName");
    const changeImageBtn    = $("changeImageBtn");
    const samplePills       = document.querySelectorAll(".sample-pill");
    const filterCards       = document.querySelectorAll(".filter-card");

    // Param panels
    const adaptiveParams    = $("adaptiveParams");
    const wienerParams      = $("wienerParams");
    const unsharpParams     = $("unsharpParams");
    const gaussianParams    = $("gaussianParams");

    // Adaptive controls
    const adaptiveStrength  = $("adaptiveStrength");
    const adaptiveStrengthVal = $("adaptiveStrengthVal");
    const adaptiveIters     = $("adaptiveIters");
    const adaptiveItersVal  = $("adaptiveItersVal");
    const adaptiveClarityCheck = $("adaptiveClarityCheck");

    // Wiener controls (new IDs matching new HTML)
    const psfTypeSelect     = $("psfTypeSelect");
    const psfSizeSlider     = $("psfSizeSlider");
    const psfSizeVal        = $("psfSizeVal");
    const psfAngleSlider    = $("psfAngleSlider");
    const psfAngleVal       = $("psfAngleVal");
    const angleRow          = $("angleRow");
    const kFactorSlider     = $("kFactorSlider");
    const kFactorVal        = $("kFactorVal");
    const autoDetectPsfBtn  = $("autoDetectPsfBtn");

    // Unsharp controls
    const unsharpAmount     = $("unsharpAmount");
    const unsharpAmountVal  = $("unsharpAmountVal");
    const unsharpRadius     = $("unsharpRadius");
    const unsharpRadiusVal  = $("unsharpRadiusVal");
    const unsharpThreshold  = $("unsharpThreshold");
    const unsharpThresholdVal = $("unsharpThresholdVal");

    // Gaussian controls
    const gaussKernel       = $("gaussKernel");
    const gaussKernelVal    = $("gaussKernelVal");
    const gaussSigma        = $("gaussSigma");
    const gaussSigmaVal     = $("gaussSigmaVal");

    // Restore button
    const restoreBtn        = $("restoreBtn");
    const restoreBtnContent = $("restoreBtnContent");
    const restoreBtnLoading = $("restoreBtnLoading");

    // Results
    const resultsPlaceholder = $("resultsPlaceholder");
    const resultsContent    = $("resultsContent");
    const techniqueBadge    = $("techniqueBadge");

    // Comparison slider
    const comparisonWrap    = $("comparisonWrap");
    const originalImgSlider = $("originalImgSlider");
    const restoredImgSlider = $("restoredImgSlider");
    const originalHalf      = $("originalHalf");
    const sliderDivider     = $("sliderDivider");

    // Side by side
    const originalImgSide   = $("originalImgSide");
    const restoredImgSide   = $("restoredImgSide");

    // View toggle
    const sliderView        = $("sliderView");
    const sideBySideView    = $("sideBySideView");
    const viewSliderBtn     = $("viewSliderBtn");
    const viewSideBySideBtn = $("viewSideBySideBtn");

    // Download
    const downloadBtn       = $("downloadBtn");
    const restoreAgainBtn   = $("restoreAgainBtn");

    // Metrics
    const metricsCard       = $("metricsCard");
    const metricAssessment  = $("metricAssessment");
    const processingTime    = $("processingTime");

    // Auth
    const navAuthLoggedOut  = $("navAuthLoggedOut");
    const navAuthLoggedIn   = $("navAuthLoggedIn");
    const navSignInBtn      = $("navSignInBtn");
    const navSignUpBtn      = $("navSignUpBtn");
    const userMenuToggle    = $("userMenuToggle");
    const userDropdownPanel = $("userDropdownPanel");
    const userNavInitials   = $("userNavInitials");
    const userNavName       = $("userNavName");
    const dropdownUserName  = $("dropdownUserName");
    const dropdownUserEmail = $("dropdownUserEmail");
    const dropdownHistoryList = $("dropdownHistoryList");
    const logoutBtn         = $("logoutBtn");

    // Auth modal
    const authModalOverlay  = $("authModalOverlay");
    const authModalClose    = $("authModalClose");
    const loginTabBtn       = $("loginTabBtn");
    const signupTabBtn      = $("signupTabBtn");
    const loginForm         = $("loginForm");
    const signupForm        = $("signupForm");
    const loginEmail        = $("loginEmail");
    const loginPassword     = $("loginPassword");
    const loginError        = $("loginError");
    const signupName        = $("signupName");
    const signupEmail       = $("signupEmail");
    const signupPassword    = $("signupPassword");
    const signupError       = $("signupError");
    const switchToSignup    = $("switchToSignup");
    const switchToLogin     = $("switchToLogin");

    /* ── Helpers ───────────────────────────────────────────────────── */
    function fmt(n) {
        if (n === undefined || n === null) return "—";
        if (n >= 1000) return (n / 1000).toFixed(1) + "k";
        return parseFloat(n).toFixed(1);
    }

    function setBarWidth(barEl, before, after) {
        const max = Math.max(before, after, 1);
        if (barEl) {
            const beforeBar = barEl.querySelector ? barEl : document.querySelector(`#${barEl.id}`);
        }
        // Used separately below
    }

    function updateMetricBars(beforeId, afterId, beforeVal, afterVal) {
        const max = Math.max(beforeVal, afterVal, 1);
        const bBar = $(beforeId);
        const aBar = $(afterId);
        if (bBar) bBar.style.width = Math.max(2, (beforeVal / max) * 100) + "%";
        if (aBar) aBar.style.width = Math.max(2, (afterVal / max) * 100) + "%";
    }

    function setMetricChange(elId, before, after) {
        const el = $(elId);
        if (!el) return;
        const pct = before > 0 ? (((after - before) / before) * 100).toFixed(1) : "—";
        const isPos = parseFloat(pct) >= 0;
        el.textContent = (isPos ? "+" : "") + pct + "%";
        el.className = "metric-change " + (isPos ? "positive" : "negative");
    }

    /* ── Filter cards ──────────────────────────────────────────────── */
    const paramPanels = {
        adaptive: adaptiveParams,
        wiener:   wienerParams,
        unsharp:  unsharpParams,
        gaussian: gaussianParams,
    };
    const paramDescs = {
        adaptive: "Auto-estimated PSF + iterative deconvolution pipeline",
        wiener:   "Adjust Point Spread Function and noise regularization",
        unsharp:  "Control sharpening strength, radius, and noise threshold",
        gaussian: "Adjust kernel size and sigma for noise smoothing",
    };

    filterCards.forEach(card => {
        card.addEventListener("click", () => {
            filterCards.forEach(c => c.classList.remove("active"));
            card.classList.add("active");
            state.activeFilter = card.dataset.filter;
            Object.values(paramPanels).forEach(p => { if (p) p.style.display = "none"; });
            const panel = paramPanels[state.activeFilter];
            if (panel) panel.style.display = "flex";
            const desc = $("paramSectionDesc");
            if (desc) desc.textContent = paramDescs[state.activeFilter] || "";
        });
    });

    // Init default: show adaptive panel
    Object.values(paramPanels).forEach(p => { if (p) p.style.display = "none"; });
    if (adaptiveParams) adaptiveParams.style.display = "flex";
    // Select adaptive card on load
    filterCards.forEach(c => {
        c.classList.toggle("active", c.dataset.filter === "adaptive");
    });

    /* ── Range bindings ─────────────────────────────────────────────── */
    function bindRange(slider, valEl, unit = "", decimals = 1) {
        if (!slider || !valEl) return;
        const update = () => valEl.textContent = parseFloat(slider.value).toFixed(decimals) + unit;
        slider.addEventListener("input", update);
        update();
    }
    bindRange(adaptiveStrength, adaptiveStrengthVal, "×", 1);
    bindRange(adaptiveIters, adaptiveItersVal, " iters", 0);
    bindRange(psfSizeSlider, psfSizeVal, " px", 0);
    bindRange(psfAngleSlider, psfAngleVal, "°", 0);
    bindRange(kFactorSlider, kFactorVal, "", 3);
    bindRange(unsharpAmount, unsharpAmountVal, "×", 1);
    bindRange(unsharpRadius, unsharpRadiusVal, " px", 1);
    bindRange(unsharpThreshold, unsharpThresholdVal, "", 0);
    bindRange(gaussKernel, gaussKernelVal, " px", 0);
    bindRange(gaussSigma, gaussSigmaVal, "", 1);

    // PSF type: show/hide angle row
    if (psfTypeSelect && angleRow) {
        const updateAngleRow = () => {
            angleRow.style.display = psfTypeSelect.value === "motion" ? "flex" : "none";
        };
        psfTypeSelect.addEventListener("change", updateAngleRow);
        updateAngleRow();
    }

    /* ── File upload ───────────────────────────────────────────────── */
    function showPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            if (sourcePreviewImg) sourcePreviewImg.src = e.target.result;
            if (previewFileName) previewFileName.textContent = file.name;
            if (dropzoneDefault) dropzoneDefault.style.display = "none";
            if (dropzonePreview) dropzonePreview.style.display = "block";
            state.uploadedFile = file;
            state.selectedSampleId = null;
            samplePills.forEach(p => p.classList.remove("active"));
            if (restoreBtn) restoreBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    if (fileInput) {
        fileInput.addEventListener("change", () => {
            if (fileInput.files[0]) showPreview(fileInput.files[0]);
        });
    }

    if (dropzone) {
        dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag-over"); });
        dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
        dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropzone.classList.remove("drag-over");
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith("image/")) showPreview(file);
        });
    }

    if (changeImageBtn) {
        changeImageBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            if (dropzoneDefault) dropzoneDefault.style.display = "flex";
            if (dropzonePreview) dropzonePreview.style.display = "none";
            state.uploadedFile = null;
            if (fileInput) fileInput.value = "";
            if (restoreBtn) restoreBtn.disabled = !state.selectedSampleId;
        });
    }

    /* ── Sample selection ──────────────────────────────────────────── */
    function selectSample(sampleId) {
        state.selectedSampleId = sampleId;
        state.uploadedFile = null;

        // Update preview
        const urls = {
            motion_blurred_license: "/static/samples/motion_blurred_license.png",
            defocus_test_pattern:   "/static/samples/defocus_test_pattern.png",
            soft_noisy_texture:     "/static/samples/soft_noisy_texture.png",
        };
        if (sourcePreviewImg) sourcePreviewImg.src = urls[sampleId] || "";
        if (previewFileName) previewFileName.textContent = sampleId.replace(/_/g, " ");
        if (dropzoneDefault) dropzoneDefault.style.display = "none";
        if (dropzonePreview) dropzonePreview.style.display = "block";

        samplePills.forEach(p => p.classList.toggle("active", p.dataset.sample === sampleId));
        if (restoreBtn) restoreBtn.disabled = false;
    }

    samplePills.forEach(pill => {
        pill.addEventListener("click", () => selectSample(pill.dataset.sample));
    });

    /* ── Hero buttons ──────────────────────────────────────────────── */
    const heroSampleBtn = $("heroSampleBtn");
    if (heroSampleBtn) {
        heroSampleBtn.addEventListener("click", () => {
            document.getElementById("samples-section")?.scrollIntoView({ behavior: "smooth" });
        });
    }

    /* ── Auto-detect PSF ───────────────────────────────────────────── */
    if (autoDetectPsfBtn) {
        autoDetectPsfBtn.addEventListener("click", async () => {
            autoDetectPsfBtn.textContent = "Analyzing...";
            autoDetectPsfBtn.disabled = true;
            try {
                const fd = new FormData();
                if (state.uploadedFile) {
                    fd.append("image", state.uploadedFile);
                } else if (state.selectedSampleId) {
                    fd.append("sample_id", state.selectedSampleId);
                } else {
                    autoDetectPsfBtn.textContent = "Auto-Detect PSF for this Image";
                    autoDetectPsfBtn.disabled = false;
                    return;
                }
                const res = await fetch("/api/estimate-blur", { method: "POST", body: fd });
                const data = await res.json();
                if (data.success && data.params) {
                    const p = data.params;
                    if (psfTypeSelect) psfTypeSelect.value = p.psf_type === "motion" ? "motion" : "gaussian";
                    if (psfSizeSlider) { psfSizeSlider.value = p.psf_size; if (psfSizeVal) psfSizeVal.textContent = p.psf_size + " px"; }
                    if (psfAngleSlider && p.psf_type === "motion") { psfAngleSlider.value = p.angle || 0; if (psfAngleVal) psfAngleVal.textContent = (p.angle || 0) + "°"; }
                    if (kFactorSlider) { kFactorSlider.value = p.k_factor; if (kFactorVal) kFactorVal.textContent = parseFloat(p.k_factor).toFixed(3); }
                    if (psfTypeSelect) psfTypeSelect.dispatchEvent(new Event("change"));
                    autoDetectPsfBtn.textContent = `Detected: ${p.blur_level} (${p.psf_type})`;
                } else {
                    autoDetectPsfBtn.textContent = "Detection failed — adjust manually";
                }
            } catch {
                autoDetectPsfBtn.textContent = "Error — try manually";
            } finally {
                autoDetectPsfBtn.disabled = false;
                setTimeout(() => { autoDetectPsfBtn.textContent = "Auto-Detect PSF for this Image"; }, 3000);
            }
        });
    }

    /* ── Restore button ────────────────────────────────────────────── */
    if (restoreBtn) {
        restoreBtn.addEventListener("click", restoreImage);
    }
    if (restoreAgainBtn) {
        restoreAgainBtn.addEventListener("click", () => {
            // Just re-enable — user can adjust filter and click restore again
            if (restoreBtnContent) restoreBtnContent.style.display = "flex";
            if (restoreBtnLoading) restoreBtnLoading.style.display = "none";
            restoreBtn.disabled = !(state.uploadedFile || state.selectedSampleId);
        });
    }

    function setRestoring(on) {
        if (!restoreBtn) return;
        restoreBtn.disabled = on;
        if (restoreBtnContent) restoreBtnContent.style.display = on ? "none" : "flex";
        if (restoreBtnLoading) restoreBtnLoading.style.display = on ? "flex" : "none";
    }

    async function restoreImage() {
        if (!state.uploadedFile && !state.selectedSampleId) return;
        setRestoring(true);

        try {
            const fd = new FormData();
            if (state.uploadedFile) {
                fd.append("image", state.uploadedFile);
            } else {
                fd.append("sample_id", state.selectedSampleId);
            }
            fd.append("filter_type", state.activeFilter);

            // Append filter-specific params
            if (state.activeFilter === "adaptive") {
                fd.append("strength", adaptiveStrength?.value || 2.2);
                fd.append("iterations", adaptiveIters?.value || 14);
                fd.append("enhance_clarity", adaptiveClarityCheck?.checked ? "true" : "false");
            } else if (state.activeFilter === "wiener") {
                fd.append("psf_type", psfTypeSelect?.value || "gaussian");
                fd.append("psf_size", psfSizeSlider?.value || 15);
                fd.append("angle", psfAngleSlider?.value || 0);
                fd.append("k_factor", kFactorSlider?.value || 0.01);
                fd.append("enhance_clarity", "true");
            } else if (state.activeFilter === "unsharp") {
                fd.append("amount", unsharpAmount?.value || 1.5);
                fd.append("radius", unsharpRadius?.value || 1.5);
                fd.append("threshold", unsharpThreshold?.value || 0);
                fd.append("enhance_clarity", "true");
            } else if (state.activeFilter === "gaussian") {
                fd.append("kernel_size", gaussKernel?.value || 5);
                fd.append("sigma", gaussSigma?.value || 1.0);
            }

            const res = await fetch("/api/restore", { method: "POST", body: fd });
            const data = await res.json();

            if (!data.success) {
                alert("Restoration failed: " + (data.error || "Unknown error"));
                setRestoring(false);
                return;
            }

            displayResults(data);
        } catch (err) {
            alert("Network error: " + err.message);
        } finally {
            setRestoring(false);
        }
    }

    /* ── Display results ───────────────────────────────────────────── */
    function displayResults(data) {
        state.originalDataUrl  = data.original_image;
        state.restoredDataUrl  = data.restored_image;
        state.downloadUrl      = data.download_url;
        state.downloadFilename = data.filename || "lumipix_restored.png";

        // Show results panel
        if (resultsPlaceholder) resultsPlaceholder.style.display = "none";
        if (resultsContent) resultsContent.style.display = "block";

        // Technique badge
        if (techniqueBadge) techniqueBadge.textContent = data.metrics?.technique_used || data.filter_type;

        // Images — slider view
        if (originalImgSlider) originalImgSlider.src = data.original_image;
        if (restoredImgSlider) restoredImgSlider.src = data.restored_image;

        // Images — side-by-side view
        if (originalImgSide) originalImgSide.src = data.original_image;
        if (restoredImgSide) restoredImgSide.src = data.restored_image;

        // Ensure slider starts at 50%
        state.splitPct = 50;
        updateSliderPosition(50);

        // Download button
        if (downloadBtn) {
            downloadBtn.href = data.download_url;
            downloadBtn.setAttribute("download", data.filename || "lumipix_restored.png");
        }

        // Metrics
        updateMetrics(data.metrics);

        // Scroll to results on mobile
        if (window.innerWidth < 1100) {
            resultsContent.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    }

    function updateMetrics(m) {
        if (!m) return;

        // Sharpness (Laplacian)
        if ($("lapBeforeVal")) $("lapBeforeVal").textContent = fmt(m.laplacian_var_before);
        if ($("lapAfterVal"))  $("lapAfterVal").textContent  = fmt(m.laplacian_var_after);
        updateMetricBars("lapBeforeBar", "lapAfterBar", m.laplacian_var_before, m.laplacian_var_after);
        setMetricChange("lapChange", m.laplacian_var_before, m.laplacian_var_after);

        // Tenengrad
        if ($("tenBeforeVal")) $("tenBeforeVal").textContent = fmt(m.tenengrad_before);
        if ($("tenAfterVal"))  $("tenAfterVal").textContent  = fmt(m.tenengrad_after);
        updateMetricBars("tenBeforeBar", "tenAfterBar", m.tenengrad_before, m.tenengrad_after);
        setMetricChange("tenChange", m.tenengrad_before, m.tenengrad_after);

        // Contrast
        if ($("conBeforeVal")) $("conBeforeVal").textContent = fmt(m.contrast_before);
        if ($("conAfterVal"))  $("conAfterVal").textContent  = fmt(m.contrast_after);
        updateMetricBars("conBeforeBar", "conAfterBar", m.contrast_before, m.contrast_after);
        setMetricChange("conChange", m.contrast_before, m.contrast_after);

        // Assessment text
        if (metricAssessment && m.assessment) metricAssessment.textContent = m.assessment;

        // Processing time
        if (processingTime && m.processing_time_ms !== undefined) {
            processingTime.textContent = `Processing time: ${m.processing_time_ms} ms`;
        }
    }

    /* ── Comparison slider drag ────────────────────────────────────── */
    function updateSliderPosition(pct) {
        state.splitPct = pct;
        if (originalHalf) originalHalf.style.width = pct + "%";
        if (sliderDivider) sliderDivider.style.left = pct + "%";
    }

    if (comparisonWrap) {
        const getX = (e) => {
            const touch = e.touches ? e.touches[0] : e;
            return touch.clientX;
        };

        const onMove = (e) => {
            if (!state.isDragging) return;
            e.preventDefault();
            const rect = comparisonWrap.getBoundingClientRect();
            const pct = Math.max(0, Math.min(100, ((getX(e) - rect.left) / rect.width) * 100));
            updateSliderPosition(pct);
        };

        comparisonWrap.addEventListener("mousedown", (e) => { state.isDragging = true; onMove(e); });
        comparisonWrap.addEventListener("touchstart", (e) => { state.isDragging = true; onMove(e); }, { passive: false });
        window.addEventListener("mousemove", onMove);
        window.addEventListener("touchmove", onMove, { passive: false });
        window.addEventListener("mouseup", () => { state.isDragging = false; });
        window.addEventListener("touchend", () => { state.isDragging = false; });
    }

    /* ── View toggle (slider ↔ side-by-side) ──────────────────────── */
    if (viewSliderBtn) {
        viewSliderBtn.addEventListener("click", () => {
            viewSliderBtn.classList.add("active");
            if (viewSideBySideBtn) viewSideBySideBtn.classList.remove("active");
            if (sliderView) sliderView.style.display = "block";
            if (sideBySideView) sideBySideView.style.display = "none";
        });
    }
    if (viewSideBySideBtn) {
        viewSideBySideBtn.addEventListener("click", () => {
            viewSideBySideBtn.classList.add("active");
            if (viewSliderBtn) viewSliderBtn.classList.remove("active");
            if (sliderView) sliderView.style.display = "none";
            if (sideBySideView) sideBySideView.style.display = "grid";
        });
    }

    /* ── Auth: show/hide user panel ───────────────────────────────── */
    function setLoggedInUser(user, history = []) {
        state.currentUser = user;
        if (navAuthLoggedOut) navAuthLoggedOut.style.display = "none";
        if (navAuthLoggedIn)  navAuthLoggedIn.style.display  = "flex";
        const initials = (user.name || "?").split(" ").map(n => n[0]).join("").slice(0, 2).toUpperCase();
        if (userNavInitials) userNavInitials.textContent = initials;
        if (userNavName) userNavName.textContent = user.name.split(" ")[0];
        if (dropdownUserName) dropdownUserName.textContent = user.name;
        if (dropdownUserEmail) dropdownUserEmail.textContent = user.email;
        if (dropdownHistoryList) {
            if (history.length === 0) {
                dropdownHistoryList.innerHTML = '<span class="empty-history-text">No recent restorations</span>';
            } else {
                dropdownHistoryList.innerHTML = history.map(h =>
                    `<div style="font-size:0.72rem;color:var(--clr-text-3);padding:0.15rem 0;">
                        ${h.filter_type} &mdash; ${h.filename?.slice(0,24) || "image"}
                    </div>`
                ).join("");
            }
        }
    }

    function setLoggedOutUser() {
        state.currentUser = null;
        if (navAuthLoggedOut) navAuthLoggedOut.style.display = "flex";
        if (navAuthLoggedIn)  navAuthLoggedIn.style.display  = "none";
    }

    async function checkCurrentUser() {
        try {
            const res = await fetch("/api/auth/me");
            const d = await res.json();
            if (d.logged_in && d.user) setLoggedInUser(d.user, d.history || []);
            else setLoggedOutUser();
        } catch { setLoggedOutUser(); }
    }

    /* ── Auth modal ────────────────────────────────────────────────── */
    function openModal(tab = "login") {
        if (!authModalOverlay) return;
        authModalOverlay.style.display = "flex";
        switchTab(tab);
        clearErrors();
    }
    function closeModal() {
        if (authModalOverlay) authModalOverlay.style.display = "none";
    }
    function switchTab(tab) {
        const isLogin = tab === "login";
        if (loginTabBtn)  loginTabBtn.classList.toggle("active", isLogin);
        if (signupTabBtn) signupTabBtn.classList.toggle("active", !isLogin);
        if (loginForm)    loginForm.style.display  = isLogin ? "flex" : "none";
        if (signupForm)   signupForm.style.display = isLogin ? "none" : "flex";
        clearErrors();
    }
    function clearErrors() {
        if (loginError)  { loginError.style.display = "none"; loginError.textContent = ""; }
        if (signupError) { signupError.style.display = "none"; signupError.textContent = ""; }
    }
    function showError(el, msg) {
        if (!el) return;
        el.textContent = msg;
        el.style.display = "block";
    }

    if (navSignInBtn) navSignInBtn.addEventListener("click", () => openModal("login"));
    if (navSignUpBtn) navSignUpBtn.addEventListener("click", () => openModal("signup"));
    if (authModalClose) authModalClose.addEventListener("click", closeModal);
    if (authModalOverlay) authModalOverlay.addEventListener("click", (e) => { if (e.target === authModalOverlay) closeModal(); });
    if (loginTabBtn) loginTabBtn.addEventListener("click", () => switchTab("login"));
    if (signupTabBtn) signupTabBtn.addEventListener("click", () => switchTab("signup"));
    if (switchToSignup) switchToSignup.addEventListener("click", (e) => { e.preventDefault(); switchTab("signup"); });
    if (switchToLogin)  switchToLogin.addEventListener("click",  (e) => { e.preventDefault(); switchTab("login"); });

    // User dropdown toggle
    if (userMenuToggle) {
        userMenuToggle.addEventListener("click", () => {
            const panel = userDropdownPanel;
            if (!panel) return;
            panel.style.display = panel.style.display === "none" ? "block" : "none";
        });
    }
    document.addEventListener("click", (e) => {
        if (userDropdownPanel && !userDropdownPanel.contains(e.target) && e.target !== userMenuToggle) {
            userDropdownPanel.style.display = "none";
        }
    });

    // Login form
    if (loginForm) {
        loginForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            clearErrors();
            try {
                const res = await fetch("/api/auth/login", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ email: loginEmail?.value, password: loginPassword?.value }),
                });
                const d = await res.json();
                if (d.success) { setLoggedInUser(d.user, []); closeModal(); }
                else showError(loginError, d.error || "Login failed. Check your email and password.");
            } catch { showError(loginError, "Network error. Please try again."); }
        });
    }

    // Signup form
    if (signupForm) {
        signupForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            clearErrors();
            if ((signupPassword?.value || "").length < 6) {
                showError(signupError, "Password must be at least 6 characters.");
                return;
            }
            try {
                const res = await fetch("/api/auth/signup", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        name: signupName?.value,
                        email: signupEmail?.value,
                        password: signupPassword?.value
                    }),
                });
                const d = await res.json();
                if (d.success) { setLoggedInUser(d.user, []); closeModal(); }
                else showError(signupError, d.error || "Sign-up failed. This email may already be registered.");
            } catch { showError(signupError, "Network error. Please try again."); }
        });
    }

    // Logout
    if (logoutBtn) {
        logoutBtn.addEventListener("click", async () => {
            await fetch("/api/auth/logout", { method: "POST" });
            setLoggedOutUser();
            if (userDropdownPanel) userDropdownPanel.style.display = "none";
        });
    }

    /* ── Navbar scroll effect ──────────────────────────────────────── */
    const navbar = document.getElementById("navbar");
    window.addEventListener("scroll", () => {
        if (navbar) navbar.style.background = window.scrollY > 20
            ? "rgba(6, 7, 26, 0.97)"
            : "rgba(6, 7, 26, 0.85)";
    });

    /* ── Init ──────────────────────────────────────────────────────── */
    checkCurrentUser();
    // Pre-load first sample preview (no auto-restore)
    selectSample("motion_blurred_license");
});
