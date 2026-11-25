(() => {
    window.__sdForgePresetsTab = window.__sdForgePresetsTab || "txt2img";

    window.sdForgePresetsActiveTab = function () {
        const root = gradioApp();
        if (root.querySelector("#tab_img2img")?.style.display === "block") {
            window.__sdForgePresetsTab = "img2img";
        } else if (root.querySelector("#tab_txt2img")?.style.display === "block") {
            window.__sdForgePresetsTab = "txt2img";
        }
        return window.__sdForgePresetsTab;
    };

    onUiTabChange(() => {
        const root = gradioApp();
        let next = null;
        if (root.querySelector("#tab_img2img")?.style.display === "block") {
            next = "img2img";
        } else if (root.querySelector("#tab_txt2img")?.style.display === "block") {
            next = "txt2img";
        }
        if (next && next !== window.__sdForgePresetsTab) {
            window.__sdForgePresetsTab = next;
            root.querySelector("#sd_forge_presets_sync_tab")?.click();
        }
    });
})();
