BeginPackage["MailLink`icons`"]
mailFolder
readMail
newMail
empty
replied
attachment
flag
Begin["`Private`"]
mailFolder = \!\(\*
GraphicsBox[
{Thickness[0.04], 
{FaceForm[{RGBColor[1., 1., 1.], Opacity[1.]}], 
       FilledCurveBox[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {
        0, 1, 0}}}, {{{20., 5.0003}, {3., 5.0003}, {3., 20.0003}, {8.,
         20.0003}, {8., 19.0003}, {20., 19.0003}}}]}, 
{RGBColor[0.392, 0.392, 0.392], Opacity[1.], JoinForm["Round"], 
       JoinedCurveBox[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {
        0, 1, 0}, {0, 1, 0}}}, {{{20., 5.0003}, {3., 5.0003}, {3., 
        20.0003}, {8., 20.0003}, {8., 19.0003}, {20., 19.0003}, {20., 
        5.0003}}},
CurveClosed->{1}]}, 
{FaceForm[{RGBColor[1., 1., 1.], Opacity[1.]}], 
       FilledCurveBox[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}}, {{{20., 
        5.0003}, {22., 17.0003}, {5., 17.0003}, {3., 5.0003}}}]}, 
{RGBColor[0.392, 0.392, 0.392], Opacity[1.], JoinForm["Round"], 
       JoinedCurveBox[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 
        0}}}, {{{20., 5.0003}, {22., 17.0003}, {5., 17.0003}, {3., 
        5.0003}, {20., 5.0003}}},
CurveClosed->{1}]}},
AspectRatio->Automatic,
ImageSize->{25., 25.},
PlotRange->{{0., 25.}, {0., 25.}}]\);

readMail = \!\(\*
GraphicsBox[
{Thickness[0.07692307692307693], 
{FaceForm[{RGBColor[1., 1., 1.], Opacity[1.]}], 
       FilledCurveBox[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 
        0}}}, {{{6.5, 5.083}, {2., 8.}, {2., 2.}, {11., 2.}, {11., 
        8.}}}]}, 
{RGBColor[0.392, 0.392, 0.392], Opacity[1.], JoinForm["Round"], 
       JoinedCurveBox[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {
        0, 1, 0}}}, {{{6.5, 5.083}, {2., 8.}, {2., 2.}, {11., 2.}, {
        11., 8.}, {6.5, 5.083}}},
CurveClosed->{1}]}, 
{FaceForm[{RGBColor[0.898, 0.898, 0.898], Opacity[1.]}], 
       FilledCurveBox[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}}, {{{6.5, 
        11.083}, {2., 8.}, {6.5, 5.083}, {11., 8.}}}]}, 
{RGBColor[0.392, 0.392, 0.392], Opacity[1.], JoinForm["Round"], 
       JoinedCurveBox[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 
        0}}}, {{{6.5, 11.083}, {2., 8.}, {6.5, 5.083}, {11., 8.}, {
        6.5, 11.083}}},
CurveClosed->{1}]}},
AspectRatio->Automatic,
ImageSize->{13., 13.},
PlotRange->{{0., 13.}, {0., 13.}}]\);

newMail = \!\(\*
GraphicsBox[
{Thickness[0.07692307692307693], 
{EdgeForm[{RGBColor[0., 0.745, 1.], Opacity[1.], Thickness[
       0.07692307692307693], CapForm["Round"], JoinForm["Round"]}], 
       FaceForm[{RGBColor[
       0.8310000000000001, 0.9490000000000001, 0.988], Opacity[1.]}], 
       FilledCurveBox[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}}, {{{11., 
        3.}, {2., 3.}, {2., 10.}, {11., 10.}}}]}, 
{RGBColor[0., 0.745, 1.], Opacity[1.], CapForm["Round"], JoinForm[
       "Round"], 
       JoinedCurveBox[{{{0, 2, 0}, {0, 1, 0}}}, {{{2., 10.}, {6.5, 
        6.083}, {11., 10.}}},
CurveClosed->{0}]}},
AspectRatio->Automatic,
ImageSize->{13., 13.},
PlotRange->{{0., 13.}, {0., 13.}}]\);

favorite = \!\(\*
GraphicsBox[
{Thickness[0.07692307692307693], FaceForm[{RGBColor[
      0.749, 0.749, 0.749], Opacity[1.]}], 
      FilledCurveBox[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0,
        1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}}}, {{{6.5, 
       3.4141}, {3.41, 1.7900999999999998`}, {4., 5.2301}, {1.5, 
       7.668099999999999}, {4.955, 8.1701}, {6.5, 11.3011}, {8.045, 
       8.1701}, {11.5, 7.668099999999999}, {9., 5.2301}, {9.59, 
       1.7900999999999998`}}}]},
AspectRatio->Automatic,
ImageSize->{13., 13.},
PlotRange->{{0., 13.}, {0., 13.}}]\);

flag = \!\(\*
GraphicsBox[
{Thickness[0.07692307692307693], 
{FaceForm[{RGBColor[0.749, 0.749, 0.749], Opacity[1.]}], 
       FilledCurveBox[{{{0, 2, 0}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}, {
        0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}, {0, 1, 
        0}, {0, 1, 0}, {0, 1, 0}}}, {{{5.837899999999999, 10.5063}, {
        8.0679, 8.6943}, {8.524899999999999, 8.3233}, {
        8.653899999999998, 8.2183}, {8.7319, 8.0713}, {8.7549, 
        7.9182999999999995`}, {9.216899999999999, 4.7393}, {
        8.918899999999999, 5.1293}, {11.499899999999998`, 
        2.5002999999999993`}, {8.0809, 3.8712999999999997`}, {8.0079, 
        3.9002999999999997`}, {7.9498999999999995`, 
        3.9532999999999996`}, {7.914899999999999, 
        4.018299999999999}, {7.7829, 4.261299999999999}, {6.2459, 
        7.0813}, {6.9319, 6.306299999999999}, {4.1179, 
        6.894299999999999}}}]}, 
{RGBColor[0.749, 0.749, 0.749], Opacity[1.], JoinForm[{"Miter", 10.}],
        JoinedCurveBox[{{{0, 2, 0}}}, {{{2., 2.}, {6., 11.}}},
CurveClosed->{0}]}},
AspectRatio->Automatic,
ImageSize->{13., 13.},
PlotRange->{{0., 13.}, {0., 13.}}]\);

replied = \!\(\*
GraphicsBox[
{Thickness[0.07692307692307693], FaceForm[{RGBColor[
      0.749, 0.749, 0.749], Opacity[1.]}], 
      FilledCurveBox[{{{0, 2, 0}, {0, 1, 0}}}, {{{1.5, 7.5}, {6.5, 
       11.5}, {6.5, 3.5}}}], 
      FilledCurveBox[{{{1, 4, 3}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0,
        1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {1, 3, 3}, {1, 3, 
       3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 
       3, 3}, {1, 3, 3}, {1, 3, 3}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {
       0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {1, 3, 
       3}}}, CompressedData["
1:eJw9U2tIVEEUvq5iVkttZmdfuo/ZbQukJGKVqPBLUTGJ0h9hVLIJRYhUVJj2
AjGRWCKiF0llZWRCSEnZExEJ06CotaiQyH7EIrthT3u6NXPn3jswnDlzzzlz
vu8711u9vWJzsqIoSXwv5tuk6IsgrQvOnLf+1CRC5ZKbg3WJAIJV90rNJoJF
XOR6sebI6W3pyXq8DxccIoGwa+uxj/v/McxNbTxZOJ3w4Rkb+ZVgWGk2ZbcQ
Yfm0V+07Jhm6St7vzVhI2JfBT78ZCkI8cj2hqe/xxaIJJm0PoWEssCgtzvCc
lzlYYwXP5iUYHpXlHV4xasXLeh4wyKCWX2fDqcJwbfQ+w4F83vGQDT1fJ1/U
dzJ842bsih1XB3hiI0NzrPyOq9mBPb1tjpyNDBXci5U7MVQnLhji4nMsE+9W
c6ARL3i3XSWbXNiiLi8EzPxbbqydsaD73LgHJ2wp/OiFoKVJ8Ui+Chha6M3T
H8NZUMPm+XB9p8h0QtATjvtxqUh0SBgV76QHZN+lszReA5pNQ66o1+8HV6O3
rdWCBIcTuKHxHJ4NQdO1sx4Nxxy4VYBug2dVt4lMnB/vGCi7TSgWz/504Etk
VbXlNSGkCmWHqFYbJXziYXlT7VKXEdLq2DDMwyvvksTZZ5W4OgiCjmCVFTUP
dh+3HSKJ8y9hqUogIfqn83PkCUndQoTLQsZ2gpperL3fQJLXIMEn5F5GaD3D
l50g2O3OIkhiCUf7v8/fMJOMuRPTmT2FjLlU+0ghY471+dV93epzr/sPPaJz
u3Ev65sNX/8//gP5Ei2u
"]]},
AspectRatio->Automatic,
ImageSize->{13., 13.},
PlotRange->{{0., 13.}, {0., 13.}}]\);

attachment = \!\(\*
GraphicsBox[
{RGBColor[0.749, 0.749, 0.749], Thickness[0.07692307692307693], 
      Opacity[1.], JoinForm[{"Miter", 10.}], 
      JoinedCurveBox[{{{0, 2, 0}, {1, 3, 3}, {1, 3, 3}, {0, 1, 0}, {1,
        3, 3}, {1, 3, 3}, {0, 1, 0}, {1, 3, 3}, {1, 3, 3}, {0, 1, 
       0}}}, CompressedData["
1:eJxTTMoPSmVmYGBgBGJxIGYCYuXp/yfUVSs7hASpL+gsEHO4U9jV9+STtMNl
7VTJRzfYHCQCb0nXJIo7MEys+21V8Nt+mvGiLeY/+OF8p4A/EsXX2eDqPaYp
9pUW/rZX5+feuuwnP5x/fx/fHOMkcQeY+s4ND19OBdpjstouPNpQ0uGcJNCi
OyoOdk2Pjs+wlnEIZl08ySpRzcHrBLvt7KkKcH6MN1BAWgmu/qGZ1IHoBcoO
e6ZN4K8KU4bz5Za/8NC7rwBXL+YBtHiznEPPk0/yl/LFHOS/5ITVvhNwYP+5
IH3zKxEHE0Ezm72X+By+g7jFQnB+WJvFtaNvBeDqF5metfa7yOewBOQhoDoY
/3za1edZr0Xg6mHh2QLk5m6RcwAaulUU6A4AYpWdAg==
"],
CurveClosed->{0}]},
AspectRatio->Automatic,
ImageSize->{13., 13.},
PlotRange->{{0., 13.}, {0., 13.}}]\);

empty = \!\(\*
GraphicsBox[
{Thickness[0.07692307692307693]},
AspectRatio->Automatic,
ImageSize->{13., 13.},
PlotRange->{{0., 13.}, {0., 13.}}]\);

elidedPlus = \!\(\*
GraphicsBox[
{Thickness[0.07692307692307693], 
{FaceForm[{RGBColor[0.749, 0.749, 0.749], Opacity[1.]}], 
       FilledCurveBox[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}}, {{{0.5, 
        0.5}, {12.5, 0.5}, {12.5, 12.5}, {0.5, 12.5}}}]}, 
{FaceForm[{RGBColor[1., 1., 1.], Opacity[1.]}], 
       FilledCurveBox[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {
        0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 
        0}, {0, 1, 0}}}, {{{10.5, 5.5}, {7.5, 5.5}, {7.5, 2.5}, {5.5, 
        2.5}, {5.5, 5.5}, {2.5, 5.5}, {2.5, 7.5}, {5.5, 7.5}, {5.5, 
        10.5}, {7.5, 10.5}, {7.5, 7.5}, {10.5, 7.5}}}]}},
AspectRatio->Automatic,
ImageSize->{13., 13.},
PlotRange->{{0., 13.}, {0., 13.}}]\);

elidedMinus = \!\(\*
GraphicsBox[
{Thickness[0.07692307692307693], 
{FaceForm[{RGBColor[0.749, 0.749, 0.749], Opacity[1.]}], 
       FilledCurveBox[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}}, {{{0.5, 
        0.5}, {12.5, 0.5}, {12.5, 12.5}, {0.5, 12.5}}}]}, 
{FaceForm[{RGBColor[1., 1., 1.], Opacity[1.]}], 
       FilledCurveBox[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}}, {{{10.5, 
        5.5}, {2.5, 5.5}, {2.5, 7.5}, {10.5, 7.5}}}]}},
AspectRatio->Automatic,
ImageSize->{13., 13.},
PlotRange->{{0., 13.}, {0., 13.}}]\);
End[] (* End Private Context *)

EndPackage[]