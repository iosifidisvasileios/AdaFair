import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import balanced_accuracy_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import numpy
import matplotlib.pyplot as plt


def plot_my_results(output_dir, dataset):
    colors = ['b','g','r','c','m','y','k', 'dimgray']

    # bank dataset
    if dataset == "bank":
        names = ['Zafar et al.', 'Krasanakis et al.', 'Adaboost', 'AFB CSB1', 'AFB CSB2']

        accuracy_list = [0.9014654090146542,0.8986906 , 0.8997750224977501, 0.8984601539846014, 0.8860663933606638]
        std_accuracy_list = [0.0020856491649241297,0.000369014633856, 0.002307587474996224, 0.0019867587972434574, 0.007645621839247798]
        balanced_accuracy_list =[0.6569178324659456, 0.64968265, 0.6884338555612595, 0.6996129078612496, 0.7499965234416308]
        std_balanced_accuracy_list =[0.004124698886348359,0.003130033274584, 0.006112057444927729, 0.010325449767000604, 0.009946341148034235]
        fairness_list =[0.026247620510718384, 0.029363, 0.11459964385212498, 0.04120110602696241, 0.05000778525975891]
        std_fairness_list =[0.013302190089989351, 0.01322239448814, 0.039373123904823705, 0.02606687327668368, 0.01200136728849429]

        tpr_protected_list =[0.3404819694085027, 0.3399602, 0.3769288688278519, 0.44688826035872264, 0.5857841060582263]
        std_tpr_protected_list= [0.013168271097689893, 0.027213980849924, 0.019418388890897493, 0.03463203990345584, 0.03198387707727941]
        tpr_non_protected_list =[0.33533764082441775, 0.3228484, 0.46463747980275044, 0.43086991392026, 0.5537558926371872]
        std_tpr_non_protected_list= [0.0170381072064758, 0.048545932917187, 0.025794234314879837, 0.026906392799051236, 0.026249547087098212]
        tnr_protected_list= [0.9772712011898972, 0.969662, 0.96924162038161381, 0.9613600652128037, 0.9312800483593888]
        std_tnr_protected_list =[0.0010575422085822227, 0.005373073189526, 0.002439124361957216, 0.005144254519056988, 0.010500738327851311]
        tnr_non_protected_list =[0.9715604016454734, 0.9798354, 0.9455251709389116, 0.9536575957373126, 0.9188569056626742]
        std_tnr_non_protected_list= [0.002640862219941835,0.005692142153882, 0.00556099856681456, 0.004932577612342491, 0.011641898432665585]

    elif dataset == "compass":
        names = ['Zafar et al.', 'Krasanakis et al.', 'AdaBoost', 'AFB CSB2']

        accuracy_list =[0.6442970822281168, 0.6709576, 0.6712769988632057, 0.657408109132247, 0.6612353164077301]
        std_accuracy_list =[0.005771835723483403,0.002795755050071, 0.010489284225590305, 0.01019240978561873, 0.01014942166480916]
        balanced_accuracy_list = [0.6357560937166155, 0.6594696875,0.6675175086392817, 0.6553809685469891, 0.6591394359804947]
        std_balanced_accuracy_list = [0.006220496647839287, 0.002896127500646,0.010979180765098788, 0.009576223767401446, 0.01024342976381424]
        fairness_list =[0.09359745167922937, 0.0424834, 0.3612745751493023, 0.07137706428828823, 0.07720410564165434]
        std_fairness_list =[0.028390909071829718, 0.019471642566238, 0.11452832362515976, 0.036535679435676836, 0.024723535414222484]

        tpr_protected_list =[0.4549621658317228, 0.5501926, 0.41987127961375315, 0.6028142435108969, 0.6059729161360606]
        std_tpr_protected_list=[0.0393259444451863, 0.112697835143546, 0.06886461753830742, 0.04954071068356159, 0.061224415765101826]
        tpr_non_protected_list =[0.49193300734626444, 0.5724758, 0.6374989253709892, 0.6244044402701291, 0.6252068340194079]
        std_tpr_non_protected_list=[0.030444830417997913, 0.13459096214682, 0.01740228200829841, 0.04567073154002051, 0.02630812486849039]
        tnr_protected_list=[0.7560185338901767, 0.7689726,  0.8408247749911053, 0.681870233122195, 0.687901038420663]
        std_tnr_protected_list =[0.032014630828735395, 0.09271311259059, 0.043274616827514024, 0.05108790299574518, 0.04843239110211022]
        tnr_non_protected_list =[0.7935183026247083, 0.76223775, 0.6971778455990392, 0.6914630325387332, 0.6980556546356925]
        std_tnr_non_protected_list=[0.03743195061831913, 0.10889897263139, 0.02245981608568674, 0.0519323490270439, 0.026752486608190986]



    elif dataset == "adult":
        names = ['Zafar et al.', 'Krasanakis et al.', 'Adaboost', 'AFB CSB1', 'AFB CSB2']

        accuracy_list =[0.8353046936807845, 0.8272768, 0.8475572663022305, 0.8438738209031525, 0.8389374380888869]
        std_accuracy_list =[0.0013164209815277018, 0.00033013739564, 0.0016006291589892912, 0.0029633063423747693, 0.004174649971287205]
        balanced_accuracy_list = [0.7096840514775167, 0.70314485, 0.7336316139467047, 0.7634373330365374, 0.7463841143306664]
        std_balanced_accuracy_list =[0.0041676454503743595, 0.002760948261015,  0.0045062591280023954, 0.011664432057266841, 0.017786380997725507]
        fairness_list = [0.13379807768266622, 0.051112,0.2228647743568891, 0.0660205275270132, 0.06269776735821179]
        std_fairness_list =[0.009987622284623092, 0.009158556474685, 0.017486347288878254, 0.01724149572624395, 0.024024282375101767]

        tpr_protected_list =[0.5591357574100587, 0.4361816, 0.3625298364804249, 0.603428470443147, 0.572569834533904]
        std_tpr_protected_list= [0.013522423527874023, 0.035273984476881, 0.018062454772540468, 0.058478616167596725, 0.0721979964258451]
        tpr_non_protected_list =[0.4427906596319713, 0.4565824,0.5330325825505013, 0.603929583075505, 0.561106127267078]
        std_tpr_non_protected_list= [0.010025543102808662, 0.049284699149844, 0.011479745452822406, 0.035357163934184456, 0.05309682666405345]
        tnr_protected_list=[0.970002907720055, 0.9436514, 0.9919610963346415, 0.9562037029373469, 0.9549031189303128]
        std_tnr_protected_list =[0.0010566979231614421, 0.02552339920701, 0.001037263342282705, 0.01144727742138612, 0.014806597032350849]
        tnr_non_protected_list = [0.9525499278154762, 0.976164,0.9395990680478288, 0.9023869875655874, 0.9144731506666506]
        std_tnr_non_protected_list=[0.003620189603987732, 0.007641821041732,0.0038272631907209716, 0.0182327472644102, 0.024449876052581615]


    elif dataset == "kdd":
        names = ['Adaboost', 'AFB CSB1', 'AFB CSB2']

        accuracy_list = [0.9439377154657796, 0.9420358520264772, 0.9043821277707194]
        std_accuracy_list =[0.0020258520531628845, 0.0009867672178378965, 0.016952192404060335]
        balanced_accuracy_list= [0.731211909432697, 0.6927326137603804, 0.7833255508413142]
        std_balanced_accuracy_list = [0.01361142454011198, 0.004562555201856117, 0.023166102374850985]
        fairness_list= [0.3680199331709092, 0.019728415478857043, 0.03855200486869861]
        std_fairness_list =  [0.03331932566098227, 0.009585460808781926, 0.026317828469936515]
        tpr_protected_list= [0.2293707996796806, 0.407130894054829, 0.6486879874722014]
        std_tpr_protected_list =[0.027443742883163647, 0.018748512021618164, 0.04809621807847681]
        tpr_non_protected_list =[0.5588181228259124, 0.4083898842579831, 0.6441365558396519]
        std_tpr_non_protected_list = [0.034462267935521707, 0.009271054163105343, 0.04127521541975133]
        tnr_protected_list =[0.9917798861541437, 0.975156762189069, 0.9190047042401519]
        std_tnr_protected_list =[0.0019635523403327393, 0.001488980492091969, 0.026293673648268158]
        tnr_non_protected_list =[0.9532072761294664, 0.9799457844558596, 0.9244748980969193]
        std_tnr_non_protected_list =[0.006891126936553181, 0.0011375249609474566, 0.017299288949667747]

        colors = ['r', 'c', 'm', 'y', 'k', 'dimgray']

    plt.figure(figsize=(9, 9))
    plt.rcParams.update({'font.size': 14})
    plt.ylim([0,1])
    plt.yticks(numpy.arange(0, 1, step=0.05))

    plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')

    plt.grid(True, axis='y')
    index = numpy.arange(0, 8, step=1.3)
    # index = numpy.arange(7)
    bar_width = 0.175



    plt.xticks(index + 1.5*bar_width ,
               ('Accuracy', 'Balanced Accuracy', 'Equalized Odds', 'TPR Prot.', 'TPR Non-Prot.', 'TNR Prot.', 'TNR Non-Prot.'))

    for i in range(0, len(names)):
        plt.bar(index + bar_width * i,
                [accuracy_list[i], balanced_accuracy_list[i], fairness_list[i], tpr_protected_list[i],
                 tpr_non_protected_list[i], tnr_protected_list[i], tnr_non_protected_list[i]], bar_width,
                yerr=[std_accuracy_list[i], std_balanced_accuracy_list[i], std_fairness_list[i],
                      std_tpr_protected_list[i], std_tpr_non_protected_list[i], std_tnr_protected_list[i],
                      std_tnr_non_protected_list[i]],
                label=names[i], color=colors[i],edgecolor='black')

    plt.legend(loc='best',ncol=1, shadow=False)
    plt.ylabel('(%)')
    # plt.title("Performance for " + dataset)
    plt.savefig(output_dir + "_performance.png",bbox_inches='tight', dpi=200)

plot_my_results("Images/kdd", "kdd")