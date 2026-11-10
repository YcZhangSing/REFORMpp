(function () {
  const INITIAL = 3;
  const STEP = 10;
  const list = document.getElementById("audit-list");
  const more = document.getElementById("audit-more");
  const collapse = document.getElementById("audit-collapse");
  const section = document.getElementById("cross-split-audit");
  if (!list || !more || !collapse || !section) return;

  let pairs = [];
  let visible = INITIAL;

  function text(tag, className, value) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    node.textContent = value;
    return node;
  }

  function sample(role, item) {
    const card = document.createElement("div");
    card.className = "audit-sample";
    card.appendChild(text("div", "audit-role", role));
    const image = document.createElement("img");
    image.className = "audit-photo";
    image.src = item.image;
    image.alt = item.sample_id;
    image.loading = "lazy";
    card.appendChild(image);
    card.appendChild(text("p", "audit-id", "[" + item.sample_id + " – " + item.domain + " – " + item.split + "]"));
    card.appendChild(text("p", "audit-kind", item.manipulation_type));
    card.appendChild(text("p", "audit-caption", item.caption));
    return card;
  }

  function card(pair) {
    const article = document.createElement("article");
    article.className = "audit-pair";
    article.appendChild(text("div", "audit-type " + pair.retrieval_type, pair.retrieval_label));
    const body = document.createElement("div");
    body.className = "audit-pair-body";
    body.appendChild(sample("Reference", pair.reference));
    const link = document.createElement("div");
    link.className = "audit-link";
    link.appendChild(text("span", "audit-sim", "Sim. " + Number(pair.similarity).toFixed(2)));
    body.appendChild(link);
    body.appendChild(sample("Query", pair.query));
    article.appendChild(body);
    return article;
  }

  function updateControls() {
    const remaining = pairs.length - visible;
    if (remaining <= 0) {
      more.hidden = true;
    } else {
      more.hidden = false;
      more.textContent = remaining < STEP ? "Show Remaining " + remaining : "Show 10 More";
    }
    collapse.hidden = visible <= INITIAL;
  }

  function render() {
    list.replaceChildren();
    pairs.slice(0, visible).forEach(function (pair) {
      list.appendChild(card(pair));
    });
    updateControls();
  }

  more.addEventListener("click", function () {
    const before = visible;
    visible = Math.min(pairs.length, visible + STEP);
    pairs.slice(before, visible).forEach(function (pair) {
      list.appendChild(card(pair));
    });
    updateControls();
  });

  collapse.addEventListener("click", function () {
    visible = INITIAL;
    render();
    section.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  fetch("assets/cross_split_audit/cross_split_demos.json?order=3", { cache: "no-store" })
    .then(function (response) {
      if (!response.ok) throw new Error("failed to load audit pairs");
      return response.json();
    })
    .then(function (data) {
      if (!Array.isArray(data) || data.length === 0) {
        throw new Error("no annotated pairs to display");
      }
      pairs = data;
      visible = Math.min(INITIAL, pairs.length);
      render();
    })
    .catch(function (error) {
      more.hidden = true;
      collapse.hidden = true;
      list.replaceChildren(text("p", "audit-count", "Annotated examples could not be loaded."));
      console.error(error);
    });
})();
