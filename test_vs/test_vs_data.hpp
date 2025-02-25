#ifndef TEST_VS_DATA_HPP
#define TEST_VS_DATA_HPP

#include "python_control.hpp"

namespace TestData {

using namespace PythonNumpy;

constexpr std::size_t SIM_SS_STEP_MAX = 50;

Matrix <DefDense, double, SIM_SS_STEP_MAX, 2> X_results_exmaple_1_answer({
    {0.1, 0.2},
    {0.21, 0.33},
    {0.313, 0.401},
    {0.3993, 0.4269},
    {0.46489, 0.42173},
    {0.509769, 0.397917},
    {0.5364217, 0.3654029},
    {0.54857577, 0.33139581},
    {0.5502822, 0.30054392},
    {0.54530632, 0.27535047},
    {0.53678452, 0.25668848},
    {0.52708686, 0.24431543},
    {0.51782389, 0.23732628},
    {0.50994198, 0.23451386},
    {0.50386216, 0.2346285} ,
    {0.49962921, 0.23654415},
    {0.49704928, 0.23934656},
    {0.4958038, 0.24236246},
    {0.49553516, 0.24514883},
    {0.49590437, 0.24745852},
    {0.49662477, 0.2491955} ,
    {0.49747644, 0.25036897},
    {0.4983073, 0.25105225},
    {0.49902556, 0.25134961},
    {0.49958781, 0.25137202},
    {0.49998587, 0.25122127},
    {0.50023436, 0.25098125},
    {0.50036031, 0.25071469},
    {0.50039515, 0.25046366},
    {0.50036934, 0.25025238},
    {0.50030901, 0.25009111},
    {0.50023453, 0.24998018},
    {0.50016021, 0.24991378},
    {0.5000949, 0.24988297},
    {0.50004302, 0.2498779} ,
    {0.5000057, 0.24988941},
    {0.49998187, 0.24990982},
    {0.49996927, 0.2499333} ,
    {0.49996515, 0.24995585},
    {0.49996678, 0.24997514},
    {0.49997177, 0.24999008},
    {0.49997826, 0.25000053},
    {0.49998488, 0.25000695},
    {0.49999081, 0.25001009},
    {0.49999558, 0.25001083},
    {0.49999908, 0.25000999},
    {0.50000135, 0.25000827},
    {0.5000026, 0.25000621},
    {0.50000306, 0.25000419},
    {0.50000298, 0.25000243}
    });


Matrix <DefDense, double, SIM_SS_STEP_MAX, 1> Y_results_exmaple_1_answer({
    {0.0},
    { 0.2 },
    {0.42},
    {0.626},
    {0.7986},
    {0.92978},
    {1.019538},
    {1.0728434},
    {1.09715154},
    {1.1005644},
    {1.09061265},
    {1.07356904},
    {1.05417372},
    {1.03564778},
    {1.01988396},
    {1.00772431},
    {0.99925842},
    {0.99409855},
    {0.99160761},
    {0.99107031},
    {0.99180875},
    {0.99324953},
    {0.99495287},
    {0.9966146},
    {0.99805112},
    {0.99917562},
    {0.99997174},
    {1.00046873},
    {1.00072061},
    {1.00079031},
    {1.00073868},
    {1.00061803},
    {1.00046906},
    {1.00032042},
    {1.00018981},
    {1.00008605},
    {1.0000114},
    {0.99996374},
    {0.99993855},
    {0.9999303},
    {0.99993355},
    {0.99994354},
    {0.99995651},
    {0.99996977},
    {0.99998162},
    {0.99999117},
    {0.99999815},
    {1.0000027},
    {1.0000052},
    {1.00000612}
    });

const std::size_t DC_MOTOR_SIM_SS_STEP_MAX = 100;

Matrix <DefDense, double, DC_MOTOR_SIM_SS_STEP_MAX, 2> Y_results_exmaple_2_answer({
    {0.0 ,  0.0 },
    {0.0 ,  0.0 },
    {0.0 ,  -0.0064009950316892 },
    {0.0 ,  -0.018550083601835302 },
    {2.56038538408566e-08 ,  -0.03582408486753065},
    {1.2515163757410706e-07 ,  -0.057627252832063006},
    {3.669995761630319e-07 ,  -0.08338939022079964},
    {8.369369097602234e-07 ,  -0.11256431705877379},
    {1.6357307857066436e-06 ,  -0.14462865250760443},
    {2.8767917703355154e-06 ,  -0.1790808717529705},
    {4.68395390175471e-06 ,  -0.2154406026987653},
    {7.189364365766353e-06 ,  -0.2532481299503569},
    {1.0531478885482765e-05 ,  -0.2920640760838908},
    {1.4853159783444608e-05 ,  -0.3314692325251667},
    {2.0299874414599704e-05 ,  -0.3710645145214858},
    {2.701799228995277e-05 ,  -0.4104710167017592},
    {3.5153179723867645e-05 ,  -0.4493301476006569},
    {4.4848891252024806e-05 ,  -0.4873038232862195},
    {5.6244957390421565e-05 ,  -0.5240747018899186},
    {6.947626854651116e-05 ,  -0.559346442404768},
    {8.467155505907495e-05 ,  -0.5928439726004123},
    {0.00010195226344072773 ,  -0.6243137523124754},
    {0.00012143152893268492 ,  -0.6535240197039918},
    {0.00014321324446182115 ,  -0.6802650093754935},
    {0.00016739122602103828 ,  -0.704349132422412},
    {0.00019404847438114572 ,  -0.7256111097080924},
    {0.00022325653289116405 ,  -0.7439080507413734},
    {0.00025507494093925717 ,  -0.7591194716221719},
    {0.00028955078243319513 ,  -0.7711472465489806},
    {0.000326718328421935 ,  -0.7799154883703461},
    {0.0003665987727229387 ,  -0.7853703546094135},
    {0.00040920005914737957 ,  -0.7874797762973609},
    {0.0004545167986313598 ,  -0.7862331078184315},
    {0.0005025302742894188 ,  -0.7816406967965658},
    {0.0005532085321104959 ,  -0.7737333738412437},
    {0.0006065065547194773 ,  -0.7625618627179049},
    {0.0006623665153326673 ,  -0.7481961122158214},
    {0.0007207181087459344 ,  -0.730724551653111},
    {0.0007814789559126928 ,  -0.7102532725841475},
    {0.0008445550783978549 ,  -0.686905139858375},
    {0.0009098414387358409 ,  -0.6608188357208864},
    {0.0009772225424778574 ,  -0.6321478411435164},
    {0.0010465730974879733 ,  -0.6010593590300854},
    {0.0011177587258408511 ,  -0.5677331843503888},
    {0.0011906367234879576 ,  -0.5323605266241376},
    {0.0012650568626951272 ,  -0.49514279049807763},
    {0.0013408622321137253 ,  -0.45629032043676254},
    {0.0014178901092314058 ,  -0.4160211157799041},
    {0.001495972859857457 ,  -0.37455952260696446},
    {0.0015749388592326435 ,  -0.3321349089929564},
    {0.001654613429314782 ,  -0.28898033033865067},
    {0.0017348197867793464 ,  -0.24533119151412253},
    {0.0018153799962892967 ,  -0.2014239125675492},
    {0.0018961159236300393 ,  -0.157494604722209},
    {0.001976850183373731 ,  -0.11377776331485645},
    {0.0020574070758316476 ,  -0.07050498421918183},
    {0.002137613508173511 ,  -0.027903710150332994},
    {0.0022172998947378323 ,  0.01380398693808127},
    {0.0022963010317265963 ,  0.05440258137269671},
    {0.002374456941670061 ,  0.09368422145599187},
    {0.002451613683261914 ,  0.1314498038842995},
    {0.0025276241224003308 ,  0.16750998974742215},
    {0.002602348660525213 ,  0.20168615730982697},
    {0.002675655916614679 ,  0.23381128713908872},
    {0.002747423359493098 ,  0.2637307755353744},
    {0.0028175378874070642 ,  0.2913031726239188},
    {0.002885896352142921 ,  0.31640084189813145},
    {0.0029524060252880745 ,  0.33891053844149477},
    {0.003016985004576489 ,  0.35873390350927803},
    {0.0030795625586046553 ,  0.3757878736135174},
    {0.003140079408556001 ,  0.3900050027240911},
    {0.003198487945927338 ,  0.40133369667231245},
    {0.0032547523856085144 ,  0.4097383593186663},
    {0.0033088488540241445 ,  0.41519945052037954},
    {0.0033607654124021146 ,  0.4177134564049143},
    {0.003410502015585737 ,  0.41729277291956945},
    {0.0034580704071530154 ,  0.4139655040827339},
    {0.003503493951945756 ,  0.40777517680642017},
    {0.003546807407441407 ,  0.3987803745903076},
    {0.0035880566357199147 ,  0.38705429280226067},
    {0.0036272982580848557 ,  0.37268421865713375},
    {0.0036645992546911733 ,  0.3557709393825217},
    {0.0037000365118095247 ,  0.3364280824151251},
    {0.003733696319618205 ,  0.3147813918028426},
    {0.003765673823656563 ,  0.29096794529383274},
    {0.0037960724332976804 ,  0.26513531687339054},
    {0.0038250031908017485 ,  0.23744068976099708},
    {0.003852584104694165 ,  0.20804992510237597},
    {0.003878939451373105 ,  0.17713659178381747},
    {0.0039041990489894858 ,  0.14488096295755426},
    {0.003928497507757313 ,  0.11146898499705227},
    {0.0039519734609440044 ,  0.07709122469929852},
    {0.003974768780858023 ,  0.0419418006170833},
    {0.003997027784195046 ,  0.0062173044380982034},
    {0.0040188964311237045 ,  -0.029884281670842228},
    {0.004040521522487986 ,  -0.06616466586640612},
    {0.004062049899475718 ,  -0.10242631975201455},
    {0.004083627650051668 ,  -0.13847353808822316},
    {0.004105399326380088 ,  -0.1741134794795025},
    {0.004127507177365617 ,  -0.20915718257864135}
    });

constexpr std::size_t SYSTEM_3_4_STEP_MAX = 71;

Matrix <DefDense, double, SYSTEM_3_4_STEP_MAX, 1> system_3_4_y_answer({
    {0.0},
    {0.0012642614672828678},
    {0.009531054189053152},
    {0.030134591902262045},
    {0.06653847940076274},
    {0.12039979547770432},
    {0.19173595883222036},
    {0.2791644511076038},
    {0.3801876329341396},
    {0.49149798919814336},
    {0.6092828482013126},
    {0.7295116419999106},
    {0.8481928607585154},
    {0.961591796789347},
    {1.0664038144947805},
    {1.1598811054712397},
    {1.23991361940819},
    {1.3050670635459922},
    {1.354582530105077},
    {1.3883434617547086},
    {1.4068163393743438},
    {1.4109717281272762},
    {1.40219221068731},
    {1.382173338653657},
    {1.352823113984678},
    {1.316164738562989},
    {1.2742465037660191},
    {1.2290617883879422},
    {1.1824812396757984},
    {1.1361983671817595},
    {1.0916890123499434},
    {1.0501844894310741},
    {1.0126576386341941},
    {0.9798205963855902},
    {0.9521327699803166},
    {0.9298172994729947},
    {0.912884189044893},
    {0.9011582810629505},
    {0.8943103144712257},
    {0.8918894399084651},
    {0.8933557417400683},
    {0.8981115272487641},
    {0.9055303718107143},
    {0.9149831437050868},
    {0.9258604626987658},
    {0.9375912640233546},
    {0.9496573370552651},
    {0.9616038810473119},
    {0.9730462655471401},
    {0.9836732991951518},
    {0.9932473973553677},
    {1.0016020976151263},
    {1.008637404659031},
    {1.014313455155228},
    {1.0186429823477752},
    {1.021683032563451},
    {1.0235263454406576},
    {1.0242927599301261},
    {1.0241209523620771},
    {1.0231607541843797},
    {1.0215662380425274},
    {1.0194897039703354},
    {1.0170766444273953},
    {1.0144617191537346},
    {1.0117657292892233},
    {1.0090935455079668},
    {1.006532917277042},
    {1.0041540696911175},
    {1.0020099803335842},
    {1.00013722074526},
    {0.9985572446703833}
});


constexpr std::size_t SYSTEM_4_4_STEP_MAX = 26;

Matrix <DefDense, double, SYSTEM_4_4_STEP_MAX, 1> system_4_4_y_answer({
    {0.9090909090909091},
    {1.7768595041322315},
    {2.113448534936138},
    {2.380643398674954},
    {2.541983595258645},
    {2.452179179830669},
    {2.3643631494792134},
    {2.352523910590856},
    {2.325248845819286},
    {2.309534515955391},
    {2.325047511755814},
    {2.33252711011215},
    {2.3309592145240994},
    {2.334614648684414},
    {2.336065699286409},
    {2.333608484094762},
    {2.3332459218484742},
    {2.3337057703048836},
    {2.3331126561674536},
    {2.333023726997603},
    {2.3333903161396274},
    {2.3333439182363924},
    {2.3332731086619414},
    {2.3333739422534374},
    {2.3333662180953954},
    {2.333315164634206}
    });

constexpr std::size_t SYSTEM_2_4_STEP_MAX = 30;

Matrix <DefDense, double, SYSTEM_2_4_STEP_MAX, 1> system_2_4_y_answer({
    {0.0},
    {0.0},
    {0.5},
    {1.7},
    {3.21},
    {4.478000000000001},
    {5.2354},
    {5.513719999999999},
    {5.464196},
    {5.234152800000001},
    {4.9377050400000035},
    {4.658833072000005},
    {4.450409729600004},
    {4.332050873280003},
    {4.2957191199040015},
    {4.317738302227201},
    {4.370703929528963},
    {4.431252829088132},
    {4.483472185643434},
    {4.519315780750817},
    {4.537296321342218},
    {4.540339671422546},
    {4.533493535944141},
    {4.522023126355022},
    {4.510179829250169},
    {4.5006665439941616},
    {4.4946475165739255},
    {4.492086969045937},
    {4.4922158843676785},
    {4.493978091095831}
    });

constexpr std::size_t SYSTEM_PID_STEP_MAX = 100;

Matrix <DefDense, double, SYSTEM_PID_STEP_MAX, 1> system_PID_y_answer({
    {0.0},
    {0.05448867675704853},
    {0.17259345313701208},
    {0.3084697601915725},
    {0.45382020030462233},
    {0.6012644765779824},
    {0.7456632944530422},
    {0.8826991841018872},
    {1.0088676111353978},
    {1.1214309009307657},
    {1.2184075016994593},
    {1.298542239836008},
    {1.36125979303678},
    {1.4066025609559978},
    {1.43515651668372},
    {1.4479686783356163},
    {1.4464597721085033},
    {1.4323354007601785},
    {1.4074986933297338},
    {1.3739670155866548},
    {1.3337948891116667},
    {1.2890048182069496},
    {1.2415272743569234},
    {1.1931506520073512},
    {1.1454815991639795},
    {1.09991575164543},
    {1.0576185683799393},
    {1.0195156822722464},
    {0.9862919500823935},
    {0.9583982066402204},
    {0.9360646029455147},
    {0.9193193320746084},
    {0.9080115178080415},
    {0.9018370539337932},
    {0.9003662319028434},
    {0.9030720750299543},
    {0.9093584025765563},
    {0.9185867706082705},
    {0.9301015724222517},
    {0.9432527238496042},
    {0.9574155025836868},
    {0.9720072511731286},
    {0.9865007864144324},
    {1.000434480257006},
    {1.013419086396973},
    {1.0251414806144243},
    {1.035365560430334},
    {1.043930610306354},
    {1.0507474824584284},
    {1.055792970999713},
    {1.0591027696065236},
    {1.0607634016058665},
    {1.0609034979777046},
    {1.059684775098607},
    {1.0572930320807097},
    {1.0539294492674793},
    {1.0498024267874797},
    {1.0451201568988087},
    {1.0400840778998153},
    {1.0348833121780965},
    {1.0296901478562743},
    {1.0246565835828794},
    {1.0199119201910223},
    {1.015561351854898},
    {1.011685483437557},
    {1.008340680153224},
    {1.0055601404803476},
    {1.0033555733023016},
    {1.0017193552195092},
    {1.0006270434482158},
    {1.0000401231832534},
    {0.9999088751757107},
    {1.0001752589429005},
    {1.0007757188536075},
    {1.0016438336891098},
    {1.002712744568278},
    {1.0039173107830468},
    {1.005195957612945},
    {1.0064921941321816},
    {1.0077557920183924},
    {1.0089436281205115},
    {1.010020203820864},
    {1.010957862882603},
    {1.0117367364261032},
    {1.0123444489083198},
    {1.0127756225253084},
    {1.0130312194066717},
    {1.0131177614483635},
    {1.013046466795246},
    {1.0128323400180013},
    {1.0124932501257793},
    {1.0120490269181415},
    {1.0115206020087089},
    {1.0109292163427304},
    {1.0102957113637703},
    {1.009639916326437},
    {1.0089801397484544},
    {1.008332768770165},
    {1.0077119763431999},
    {1.007129532779219}
    });


    constexpr std::size_t LKF_SIM_STEP_MAX = 50;

    Matrix <DefDense, double, LKF_SIM_STEP_MAX, 2> lkf_test_input({
        { 0.5, 0.5 },
        { 0.5, 0.5 },
        { -0.5, -0.5 },
        { -0.5, 0.5 },
        { -0.5, -0.5 },
        { 0.5, 0.5 },
        { -0.5, 0.5 },
        { -0.5, 0.5 },
        { 0.5, 0.5 },
        { 0.5, -0.5 },
        { -0.5, -0.5 },
        { 0.5, -0.5 },
        { -0.5, 0.5 },
        { 0.5, -0.5 },
        { 0.5, -0.5 },
        { 0.5, 0.5 },
        { 0.5, 0.5 },
        { -0.5, -0.5 },
        { -0.5, 0.5 },
        { -0.5, -0.5 },
        { 0.5, 0.5 },
        { -0.5, 0.5 },
        { -0.5, 0.5 },
        { 0.5, 0.5 },
        { 0.5, -0.5 },
        { -0.5, -0.5 },
        { 0.5, -0.5 },
        { -0.5, 0.5 },
        { 0.5, -0.5 },
        { 0.5, -0.5 },
        { 0.5, 0.5 },
        { 0.5, 0.5 },
        { -0.5, -0.5 },
        { -0.5, 0.5 },
        { -0.5, -0.5 },
        { 0.5, 0.5 },
        { -0.5, 0.5 },
        { -0.5, 0.5 },
        { 0.5, 0.5 },
        { 0.5, -0.5 },
        { -0.5, -0.5 },
        { 0.5, -0.5 },
        { -0.5, 0.5 },
        { 0.5, -0.5 },
        { 0.5, -0.5 },
        { 0.5, 0.5 },
        { 0.5, 0.5 },
        { -0.5, -0.5 },
        { -0.5, 0.5 },
        { -0.5, -0.5 }
        });


} // namespace TestData

#endif // TEST_VS_DATA_HPP