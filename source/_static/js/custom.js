(() => {
      const application = Stimulus.Application.start();

      class NotebookToggle extends Stimulus.Controller {
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

      application.register("notebook-toggle", NotebookToggle)
})()
