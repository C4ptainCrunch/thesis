import {Controller, Application} from 'https://cdn.skypack.dev/stimulus';

const application = Application.start();

class NotebookToggle extends Controller {
  static targets = [ "cell"]
  static values = { show: Boolean }

  toggle() {
    this.showValue = !this.showValue;

    this.cellTargets.forEach((el, i) => {
        el.classList.toggle("filter--filtered", !this.showValue)
    })
  }

  connect() {
    document.querySelectorAll(".highlight-ipython3").forEach((e) => e.setAttribute("data-notebook-toggle-target", "cell"))
    document.querySelectorAll(".highlight-none").forEach((e) => e.setAttribute("data-notebook-toggle-target", "cell"))
  }
}

class Pseudocode extends Controller {
  connect() {
    pseudocode.renderElement(this.element, {noEnd: true,});

  }
}

application.register("notebook-toggle", NotebookToggle)

var load = function() {
  console.log("onLoad detected pseudocode");
  application.register("pseudocode", Pseudocode)
}

var script = document.querySelector('#pseudocode-script');
script.addEventListener('load', load);

var time = function() {
  if(pseudocode !== undefined)
  {
    console.log("Polling loop detected pseudocode")
    application.register("pseudocode", Pseudocode)
  }
  else setTimeout(time, 500)
}

time();
