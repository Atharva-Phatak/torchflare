::: torchflare.callbacks.notifiers.message_notifiers.DiscordNotifierCallback
    rendering:
         show_root_toc_entry: false

## Examples

``` python
import torchflare.callbacks as cbs

discord_notif = cbs.DiscordNotifierCallback(
    webhook_url="YOUR_SECRET_URL", exp_name="MODEL_RUN"
)
```
