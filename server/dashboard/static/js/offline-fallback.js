// Offline dashboard fallback.
// The field router may not have internet access, so CDN scripts can fail.
// Keep the dashboard usable enough for packet/status checks in that case.
(function () {
    if (!window.io) {
        window.io = function () {
            return {
                connected: false,
                on: function () {},
                emit: function () {}
            };
        };
    }

    if (!window.Chart) {
        window.Chart = class {
            constructor(_ctx, config) {
                this.type = config && config.type;
                this.data = (config && config.data) || { labels: [], datasets: [] };
                this.options = (config && config.options) || {};
            }

            update() {}
        };
    }
})();
