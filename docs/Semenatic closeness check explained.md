## Semenatic closeness check - cosine similarity

The dot product, `np.dot(emb_source, emb_output)`, measures how much the two vectors point in the same direction; dividing by the product of their lengths normalizes the result. The cos value is typically between -1 and 1, with values closer to 1 indicating greater semantic similarity.

The dot product multiplies corresponding entries in the two embedding vectors and adds the results. For example, the dot product of `[1, 2]` and `[3, 4]` is:

`1 × 3 + 2 × 4 = 11`

In this code, `np.dot(emb_source, emb_output)` applies that operation to the vectors representing the source text and the model’s answer.

Geometrically, the dot product reflects both how aligned the vectors are and how large they are. A positive value means they point generally in the same direction; a negative value means they point in opposing directions; and a value near zero means they are roughly perpendicular.

Because vector length also affects the dot product, the code divides by both vectors’ lengths. That normalization leaves cosine similarity, which focuses on directional alignment rather than raw vector size.