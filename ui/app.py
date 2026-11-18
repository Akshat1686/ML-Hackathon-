"""PathSnap evaluator UI  (runs on LAPTOP 2).

Start:
    streamlit run ui/app.py
Then enter the API address of Laptop 1, e.g. http://192.168.1.23:8000
The UI only talks to the API: it never reads trip files, models or reference routes.
"""
import requests
import streamlit as st

st.set_page_config(page_title="PathSnap", page_icon="🛰️", layout="wide")
st.title("🛰️ PathSnap — GPS map-matching & route reconstruction")

api = st.sidebar.text_input("API URL (Laptop 1)", value=st.session_state.get("api", "http://127.0.0.1:8000"))
st.session_state.api = api.rstrip("/")
show_clf = st.sidebar.checkbox("Also show classifier-only route", value=False)

try:
    health = requests.get(f"{st.session_state.api}/health", timeout=5).json()
    st.sidebar.success(f"API online · {health.get('model')}")
    trips = requests.get(f"{st.session_state.api}/trips", timeout=10).json()["trips"]
except Exception as e:  # noqa: BLE001
    st.error(f"Cannot reach API at {st.session_state.api}: {e}")
    st.stop()

if not trips:
    st.warning("The API has no evaluation trips in its data/evaluator_test folder.")
    st.stop()

st.subheader(f"Evaluation trips ({len(trips)})")
trip_id = st.radio("Select a trip", trips, horizontal=True)

if st.button("▶ Reconstruct route", type="primary"):
    with st.spinner(f"Sending trip {trip_id} to the API..."):
        r = requests.post(f"{st.session_state.api}/predict", json={"trip_id": trip_id}, timeout=600)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()
    st.session_state.result = r.json()

res = st.session_state.get("result")
if res:
    st.markdown(f"### Trip `{res['trip_id']}`")
    m = res.get("metrics")
    if m:
        c = st.columns(5)
        c[0].metric("Top-1 accuracy (HMM)", f"{m['hmm']['top1_accuracy']*100:.2f}%",
                    f"{(m['hmm']['top1_accuracy']-m['classifier_only']['top1_accuracy'])*100:+.2f} vs classifier")
        c[1].metric("Route continuity", f"{m['hmm']['route_continuity']*100:.1f}%")
        c[2].metric("Route F1", f"{m['hmm']['route_f1']*100:.1f}%")
        c[3].metric("Route precision / recall", f"{m['hmm']['route_precision']*100:.0f}% / {m['hmm']['route_recall']*100:.0f}%")
        c[4].metric("API time", f"{res['timing_s']['request_total']:.1f}s")
        with st.expander("All metrics (computed server-side)"):
            st.json(m)
    else:
        st.info("No reference route on the server for this trip — metrics unavailable.")
    st.caption(f"{len(res['gps'])} GPS observations · {len(res['predicted_route'])} matched road segments · "
               f"HMM segments: {res['hmm_stats']['n_segments']}")

    try:
        import folium
        import streamlit.components.v1 as components

        gps = res["gps"]
        fmap = folium.Map(location=gps[len(gps) // 2], zoom_start=14, tiles="OpenStreetMap")
        fg_gps = folium.FeatureGroup("Raw GPS observations")
        for lat, lon in gps[::2]:
            folium.CircleMarker([lat, lon], radius=2, color="#1f77b4", fill=True, fill_opacity=.7).add_to(fg_gps)
        fg_gps.add_to(fmap)
        fg_pred = folium.FeatureGroup("Predicted route (classifier + HMM)")
        for seg in res["predicted_route"]:
            folium.PolyLine(seg, color="#d62728", weight=5, opacity=.85).add_to(fg_pred)
        fg_pred.add_to(fmap)
        if show_clf:
            fg_c = folium.FeatureGroup("Classifier-only route", show=True)
            for seg in res["classifier_only_route"]:
                folium.PolyLine(seg, color="#ff7f0e", weight=3, opacity=.7, dash_array="6").add_to(fg_c)
            fg_c.add_to(fmap)
        folium.Marker(gps[0], tooltip="start", icon=folium.Icon(color="green")).add_to(fmap)
        folium.Marker(gps[-1], tooltip="end", icon=folium.Icon(color="black")).add_to(fmap)
        lats = [p[0] for p in gps]
        lons = [p[1] for p in gps]
        fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
        folium.LayerControl(collapsed=False).add_to(fmap)
        legend = ("<div style='position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px;"
                  "border:1px solid #999;font-size:13px'><b>Legend</b><br>"
                  "<span style='color:#1f77b4'>●</span> raw GPS<br><span style='color:#d62728'>━</span> predicted route"
                  + ("<br><span style='color:#ff7f0e'>┅</span> classifier-only" if show_clf else "") + "</div>")
        fmap.get_root().html.add_child(folium.Element(legend))
        components.html(fmap.get_root().render(), height=620)
    except Exception as e:  # noqa: BLE001  (fallback if folium/tiles unavailable offline)
        import matplotlib.pyplot as plt
        st.warning(f"Interactive map unavailable ({e}); showing static plot.")
        fig, ax = plt.subplots(figsize=(8, 8))
        g = res["gps"]
        ax.scatter([p[1] for p in g], [p[0] for p in g], s=3, label="raw GPS")
        for i, seg in enumerate(res["predicted_route"]):
            ax.plot([seg[0][1], seg[1][1]], [seg[0][0], seg[1][0]], color="red", lw=2,
                    label="predicted route" if i == 0 else None)
        ax.legend()
        ax.set_aspect("equal")
        st.pyplot(fig)
