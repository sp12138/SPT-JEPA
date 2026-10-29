# Decoder side information can hide task-relevant information

Toy example: X=(C,M) contains two independent scene variables. The decoder also observes anchor A=C. A code Z=M permits exact reconstruction of X from (Z,A) while discarding C completely. Z varies and can pass a noncollapse test, yet a goal/control problem requiring C is impossible from Z alone. Thus conditional reconstruction does not by itself guarantee a control-sufficient representation.

In the current architecture, the planner uses the encoder/prior code, while the renderer receives a past-image anchor. This makes the toy mechanism relevant as a hypothesis; it does not establish that real PushT errors are caused by omitted scene information. Limited early-window motion might let the decoder recover a block from the anchor without paying to encode its complete current state.

Randomly dropping the own anchor changes which information the encoder must supply. It may improve control information, or it may spend capacity on irrelevant details and hurt planning. Three probabilities test that balance. The context-mask variant uses only the first observed frame versus the past anchor, suppressing changed regions; future target frames never construct this mask.

All operations stay inside each video sample, with private random masks preserving posterior/flow draw order. No trainable parameter is added. Normal past anchors are retained at inference. This is a bounded600updateadaptation screen; successful adaptation would still require unused evaluation cases and from-scratch training confirmation.
