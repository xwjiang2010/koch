package main

import (
	"regexp"
	"strings"
	"testing"

	"github.com/gruntwork-io/terratest/modules/helm"
	"github.com/gruntwork-io/terratest/modules/k8s"
	"github.com/gruntwork-io/terratest/modules/random"
	"github.com/stretchr/testify/require"
	appsV1 "k8s.io/api/apps/v1"
	coreV1 "k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	netV1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

const (
	chartName     = "auto-deploy-app-2.0.1"
	helmChartPath = ".."
)

func TestDeploymentTemplate(t *testing.T) {
	for _, tc := range []struct {
		CaseName string
		Release  string
		Values   map[string]string

		ExpectedErrorRegexp *regexp.Regexp

		ExpectedName         string
		ExpectedRelease      string
		ExpectedStrategyType appsV1.DeploymentStrategyType
	}{
		{
			CaseName: "happy",
			Release:  "production",
			Values: map[string]string{
				"releaseOverride": "productionOverridden",
			},
			ExpectedName:         "productionOverridden",
			ExpectedRelease:      "production",
			ExpectedStrategyType: appsV1.DeploymentStrategyType(""),
		}, {
			// See https://github.com/helm/helm/issues/6006
			CaseName: "long release name",
			Release:  strings.Repeat("r", 80),

			ExpectedErrorRegexp: regexp.MustCompile("Error: release name .* exceeds max length of 53"),
		},
		{
			CaseName: "strategyType",
			Release:  "production",
			Values: map[string]string{
				"strategyType": "Recreate",
			},
			ExpectedName:         "production",
			ExpectedRelease:      "production",
			ExpectedStrategyType: appsV1.RecreateDeploymentStrategyType,
		},
	} {
		t.Run(tc.CaseName, func(t *testing.T) {
			namespaceName := "minimal-ruby-app-" + strings.ToLower(random.UniqueId())

			values := map[string]string{
				"gitlab.app": "auto-devops-examples/minimal-ruby-app",
				"gitlab.env": "prod",
			}

			mergeStringMap(values, tc.Values)

			options := &helm.Options{
				SetValues:      values,
				KubectlOptions: k8s.NewKubectlOptions("", "", namespaceName),
			}

			output, err := helm.RenderTemplateE(t, options, helmChartPath, tc.Release, []string{"templates/deployment.yaml"})

			if tc.ExpectedErrorRegexp != nil {
				require.Regexp(t, tc.ExpectedErrorRegexp, err.Error())
				return
			}
			if err != nil {
				t.Error(err)
				return
			}

			var deployment appsV1.Deployment
			helm.UnmarshalK8SYaml(t, output, &deployment)

			require.Equal(t, tc.ExpectedName, deployment.Name)
			require.Equal(t, tc.ExpectedStrategyType, deployment.Spec.Strategy.Type)

			require.Equal(t, map[string]string{
				"app.gitlab.com/app": "auto-devops-examples/minimal-ruby-app",
				"app.gitlab.com/env": "prod",
			}, deployment.Annotations)
			require.Equal(t, map[string]string{
				"app":      tc.ExpectedName,
				"chart":    chartName,
				"heritage": "Helm",
				"release":  tc.ExpectedRelease,
				"tier":     "web",
				"track":    "stable",
			}, deployment.Labels)

			require.Equal(t, map[string]string{
				"app.gitlab.com/app":           "auto-devops-examples/minimal-ruby-app",
				"app.gitlab.com/env":           "prod",
				"checksum/application-secrets": "",
			}, deployment.Spec.Template.Annotations)
			require.Equal(t, map[string]string{
				"app":     tc.ExpectedName,
				"release": tc.ExpectedRelease,
				"tier":    "web",
				"track":   "stable",
			}, deployment.Spec.Template.Labels)
		})
	}

	for _, tc := range []struct {
		CaseName                string
		Release                 string
		Values                  map[string]string
		ExpectedImageRepository string
	}{
		{
			CaseName: "skaffold",
			Release:  "production",
			Values: map[string]string{
				"image.repository": "skaffold",
				"image.tag":        "",
			},
			ExpectedImageRepository: "skaffold",
		},
		{
			CaseName: "skaffold",
			Release:  "production",
			Values: map[string]string{
				"image.repository": "skaffold",
				"image.tag":        "stable",
			},
			ExpectedImageRepository: "skaffold:stable",
		},
	} {
		t.Run(tc.CaseName, func(t *testing.T) {
			namespaceName := "minimal-ruby-app-" + strings.ToLower(random.UniqueId())

			values := map[string]string{
				"gitlab.app": "auto-devops-examples/minimal-ruby-app",
				"gitlab.env": "prod",
			}

			mergeStringMap(values, tc.Values)

			options := &helm.Options{
				SetValues:      values,
				KubectlOptions: k8s.NewKubectlOptions("", "", namespaceName),
			}

			output := helm.RenderTemplate(t, options, helmChartPath, tc.Release, []string{"templates/deployment.yaml"})

			var deployment appsV1.Deployment
			helm.UnmarshalK8SYaml(t, output, &deployment)

			require.Equal(t, tc.ExpectedImageRepository, deployment.Spec.Template.Spec.Containers[0].Image)
		})
	}

	// deployment livenessProbe, and readinessProbe tests
	for _, tc := range []struct {
		CaseName string
		Release  string
		Values   map[string]string

		ExpectedLivenessProbe  *coreV1.Probe
		ExpectedReadinessProbe *coreV1.Probe
	}{
		{
			CaseName:               "defaults",
			Release:                "production",
			ExpectedLivenessProbe:  defaultLivenessProbe(),
			ExpectedReadinessProbe: defaultReadinessProbe(),
		},
	} {
		t.Run(tc.CaseName, func(t *testing.T) {
			namespaceName := "minimal-ruby-app-" + strings.ToLower(random.UniqueId())

			values := map[string]string{
				"gitlab.app": "auto-devops-examples/minimal-ruby-app",
				"gitlab.env": "prod",
			}

			mergeStringMap(values, tc.Values)

			options := &helm.Options{
				SetValues:      values,
				KubectlOptions: k8s.NewKubectlOptions("", "", namespaceName),
			}

			output := helm.RenderTemplate(t, options, helmChartPath, tc.Release, []string{"templates/deployment.yaml"})

			var deployment appsV1.Deployment
			helm.UnmarshalK8SYaml(t, output, &deployment)

			require.Equal(t, tc.ExpectedLivenessProbe, deployment.Spec.Template.Spec.Containers[0].LivenessProbe)
			require.Equal(t, tc.ExpectedReadinessProbe, deployment.Spec.Template.Spec.Containers[0].ReadinessProbe)
		})
	}

	// Test Deployment selector
	for _, tc := range []struct {
		CaseName string
		Release  string
		Values   map[string]string

		ExpectedName     string
		ExpectedRelease  string
		ExpectedSelector *metav1.LabelSelector
	}{
		{
			CaseName:        "selector",
			Release:         "production",
			ExpectedName:    "production",
			ExpectedRelease: "production",
			ExpectedSelector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"app":     "production",
					"release": "production",
					"tier":    "web",
					"track":   "stable",
				},
			},
		},
	} {
		t.Run(tc.CaseName, func(t *testing.T) {
			namespaceName := "minimal-ruby-app-" + strings.ToLower(random.UniqueId())

			values := map[string]string{
				"gitlab.app": "auto-devops-examples/minimal-ruby-app",
				"gitlab.env": "prod",
			}

			mergeStringMap(values, tc.Values)

			options := &helm.Options{
				SetValues:      values,
				KubectlOptions: k8s.NewKubectlOptions("", "", namespaceName),
			}

			output := helm.RenderTemplate(t, options, helmChartPath, tc.Release, []string{"templates/deployment.yaml"})

			var deployment appsV1.Deployment
			helm.UnmarshalK8SYaml(t, output, &deployment)

			require.Equal(t, tc.ExpectedName, deployment.Name)
			require.Equal(t, map[string]string{
				"app":      tc.ExpectedName,
				"chart":    chartName,
				"heritage": "Helm",
				"release":  tc.ExpectedRelease,
				"tier":     "web",
				"track":    "stable",
			}, deployment.Labels)

			require.Equal(t, tc.ExpectedSelector, deployment.Spec.Selector)

			require.Equal(t, map[string]string{
				"app":     tc.ExpectedName,
				"release": tc.ExpectedRelease,
				"tier":    "web",
				"track":   "stable",
			}, deployment.Spec.Template.Labels)
		})
	}
}

func TestWorkerDeploymentTemplate(t *testing.T) {
	for _, tc := range []struct {
		CaseName string
		Release  string
		Values   map[string]string

		ExpectedErrorRegexp *regexp.Regexp

		ExpectedName        string
		ExpectedRelease     string
		ExpectedDeployments []workerDeploymentTestCase
	}{
		{
			CaseName: "happy",
			Release:  "production",
			Values: map[string]string{
				"releaseOverride":            "productionOverridden",
				"workers.worker1.command[0]": "echo",
				"workers.worker1.command[1]": "worker1",
				"workers.worker2.command[0]": "echo",
				"workers.worker2.command[1]": "worker2",
			},
			ExpectedName:    "productionOverridden",
			ExpectedRelease: "production",
			ExpectedDeployments: []workerDeploymentTestCase{
				{
					ExpectedName:         "productionOverridden-worker1",
					ExpectedCmd:          []string{"echo", "worker1"},
					ExpectedStrategyType: appsV1.DeploymentStrategyType(""),
				},
				{
					ExpectedName:         "productionOverridden-worker2",
					ExpectedCmd:          []string{"echo", "worker2"},
					ExpectedStrategyType: appsV1.DeploymentStrategyType(""),
				},
			},
		}, {
			// See https://github.com/helm/helm/issues/6006
			CaseName: "long release name",
			Release:  strings.Repeat("r", 80),

			ExpectedErrorRegexp: regexp.MustCompile("Error: release name .* exceeds max length of 53"),
		},
		{
			CaseName: "strategyType",
			Release:  "production",
			Values: map[string]string{
				"workers.worker1.command[0]":   "echo",
				"workers.worker1.command[1]":   "worker1",
				"workers.worker1.strategyType": "Recreate",
			},
			ExpectedName:    "production",
			ExpectedRelease: "production",
			ExpectedDeployments: []workerDeploymentTestCase{
				{
					ExpectedName:         "production" + "-worker1",
					ExpectedCmd:          []string{"echo", "worker1"},
					ExpectedStrategyType: appsV1.RecreateDeploymentStrategyType,
				},
			},
		},
	} {
		t.Run(tc.CaseName, func(t *testing.T) {
			namespaceName := "minimal-ruby-app-" + strings.ToLower(random.UniqueId())

			values := map[string]string{
				"gitlab.app": "auto-devops-examples/minimal-ruby-app",
				"gitlab.env": "prod",
			}

			mergeStringMap(values, tc.Values)

			options := &helm.Options{
				SetValues:      values,
				KubectlOptions: k8s.NewKubectlOptions("", "", namespaceName),
			}

			output, err := helm.RenderTemplateE(t, options, helmChartPath, tc.Release, []string{"templates/worker-deployment.yaml"})

			if tc.ExpectedErrorRegexp != nil {
				require.Regexp(t, tc.ExpectedErrorRegexp, err.Error())
				return
			}
			if err != nil {
				t.Error(err)
				return
			}

			var deployments deploymentList
			helm.UnmarshalK8SYaml(t, output, &deployments)

			require.Len(t, deployments.Items, len(tc.ExpectedDeployments))
			for i, expectedDeployment := range tc.ExpectedDeployments {
				deployment := deployments.Items[i]

				require.Equal(t, expectedDeployment.ExpectedName, deployment.Name)
				require.Equal(t, expectedDeployment.ExpectedStrategyType, deployment.Spec.Strategy.Type)

				require.Equal(t, map[string]string{
					"app.gitlab.com/app": "auto-devops-examples/minimal-ruby-app",
					"app.gitlab.com/env": "prod",
				}, deployment.Annotations)
				require.Equal(t, map[string]string{
					"chart":    chartName,
					"heritage": "Helm",
					"release":  tc.ExpectedRelease,
					"tier":     "worker",
					"track":    "stable",
				}, deployment.Labels)

				require.Equal(t, map[string]string{
					"app.gitlab.com/app":           "auto-devops-examples/minimal-ruby-app",
					"app.gitlab.com/env":           "prod",
					"checksum/application-secrets": "",
				}, deployment.Spec.Template.Annotations)
				require.Equal(t, map[string]string{
					"release": tc.ExpectedRelease,
					"tier":    "worker",
					"track":   "stable",
				}, deployment.Spec.Template.Labels)

				require.Len(t, deployment.Spec.Template.Spec.Containers, 1)
				require.Equal(t, expectedDeployment.ExpectedCmd, deployment.Spec.Template.Spec.Containers[0].Command)
			}
		})
	}

	// Tests worker selector
	for _, tc := range []struct {
		CaseName string
		Release  string
		Values   map[string]string

		ExpectedName        string
		ExpectedRelease     string
		ExpectedDeployments []workerDeploymentSelectorTestCase
	}{
		{
			CaseName: "worker selector",
			Release:  "production",
			Values: map[string]string{
				"workers.worker1.command[0]": "echo",
				"workers.worker1.command[1]": "worker1",
				"workers.worker2.command[0]": "echo",
				"workers.worker2.command[1]": "worker2",
			},
			ExpectedName:    "production",
			ExpectedRelease: "production",
			ExpectedDeployments: []workerDeploymentSelectorTestCase{
				{
					ExpectedName: "production-worker1",
					ExpectedSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"release": "production",
							"tier":    "worker",
							"track":   "stable",
						},
					},
				},
				{
					ExpectedName: "production-worker2",
					ExpectedSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"release": "production",
							"tier":    "worker",
							"track":   "stable",
						},
					},
				},
			},
		},
	} {
		t.Run(tc.CaseName, func(t *testing.T) {
			namespaceName := "minimal-ruby-app-" + strings.ToLower(random.UniqueId())

			values := map[string]string{
				"gitlab.app": "auto-devops-examples/minimal-ruby-app",
				"gitlab.env": "prod",
			}

			mergeStringMap(values, tc.Values)

			options := &helm.Options{
				SetValues:      values,
				KubectlOptions: k8s.NewKubectlOptions("", "", namespaceName),
			}

			output := helm.RenderTemplate(t, options, helmChartPath, tc.Release, []string{"templates/worker-deployment.yaml"})

			var deployments deploymentAppsV1List
			helm.UnmarshalK8SYaml(t, output, &deployments)

			require.Len(t, deployments.Items, len(tc.ExpectedDeployments))
			for i, expectedDeployment := range tc.ExpectedDeployments {
				deployment := deployments.Items[i]

				require.Equal(t, expectedDeployment.ExpectedName, deployment.Name)

				require.Equal(t, map[string]string{
					"chart":    chartName,
					"heritage": "Helm",
					"release":  tc.ExpectedRelease,
					"tier":     "worker",
					"track":    "stable",
				}, deployment.Labels)

				require.Equal(t, expectedDeployment.ExpectedSelector, deployment.Spec.Selector)

				require.Equal(t, map[string]string{
					"release": tc.ExpectedRelease,
					"tier":    "worker",
					"track":   "stable",
				}, deployment.Spec.Template.Labels)
			}
		})
	}

	// worker livenessProbe, and readinessProbe tests
	for _, tc := range []struct {
		CaseName string
		Values   map[string]string
		Release  string

		ExpectedDeployments []workerDeploymentTestCase
	}{
		{
			CaseName: "default liveness and readiness values",
			Release:  "production",
			Values: map[string]string{
				"workers.worker1.command[0]": "echo",
				"workers.worker1.command[1]": "worker1",
				"workers.worker2.command[0]": "echo",
				"workers.worker2.command[1]": "worker2",
			},
			ExpectedDeployments: []workerDeploymentTestCase{
				{
					ExpectedName:           "production-worker1",
					ExpectedCmd:            []string{"echo", "worker1"},
					ExpectedLivenessProbe:  defaultLivenessProbe(),
					ExpectedReadinessProbe: defaultReadinessProbe(),
				},
				{
					ExpectedName:           "production-worker2",
					ExpectedCmd:            []string{"echo", "worker2"},
					ExpectedLivenessProbe:  defaultLivenessProbe(),
					ExpectedReadinessProbe: defaultReadinessProbe(),
				},
			},
		},
		{
			CaseName: "enableWorkerLivenessProbe",
			Release:  "production",
			Values: map[string]string{
				"workers.worker1.command[0]":              "echo",
				"workers.worker1.command[1]":              "worker1",
				"workers.worker1.livenessProbe.path":      "/worker",
				"workers.worker1.livenessProbe.scheme":    "HTTP",
				"workers.worker1.livenessProbe.probeType": "httpGet",
				"workers.worker2.command[0]":              "echo",
				"workers.worker2.command[1]":              "worker2",
				"workers.worker2.livenessProbe.path":      "/worker",
				"workers.worker2.livenessProbe.scheme":    "HTTP",
				"workers.worker2.livenessProbe.probeType": "httpGet",
			},
			ExpectedDeployments: []workerDeploymentTestCase{
				{
					ExpectedName:           "production-worker1",
					ExpectedCmd:            []string{"echo", "worker1"},
					ExpectedLivenessProbe:  workerLivenessProbe(),
					ExpectedReadinessProbe: defaultReadinessProbe(),
				},
				{
					ExpectedName:           "production-worker2",
					ExpectedCmd:            []string{"echo", "worker2"},
					ExpectedLivenessProbe:  workerLivenessProbe(),
					ExpectedReadinessProbe: defaultReadinessProbe(),
				},
			},
		},
		{
			CaseName: "enableWorkerReadinessProbe",
			Release:  "production",
			Values: map[string]string{
				"workers.worker1.command[0]":               "echo",
				"workers.worker1.command[1]":               "worker1",
				"workers.worker1.readinessProbe.path":      "/worker",
				"workers.worker1.readinessProbe.scheme":    "HTTP",
				"workers.worker1.readinessProbe.probeType": "httpGet",
				"workers.worker2.command[0]":               "echo",
				"workers.worker2.command[1]":               "worker2",
				"workers.worker2.readinessProbe.path":      "/worker",
				"workers.worker2.readinessProbe.scheme":    "HTTP",
				"workers.worker2.readinessProbe.probeType": "httpGet",
			},
			ExpectedDeployments: []workerDeploymentTestCase{
				{
					ExpectedName:           "production-worker1",
					ExpectedCmd:            []string{"echo", "worker1"},
					ExpectedLivenessProbe:  defaultLivenessProbe(),
					ExpectedReadinessProbe: workerReadinessProbe(),
				},
				{
					ExpectedName:           "production-worker2",
					ExpectedCmd:            []string{"echo", "worker2"},
					ExpectedLivenessProbe:  defaultLivenessProbe(),
					ExpectedReadinessProbe: workerReadinessProbe(),
				},
			},
		},
	} {
		t.Run(tc.CaseName, func(t *testing.T) {
			namespaceName := "minimal-ruby-app-" + strings.ToLower(random.UniqueId())

			values := map[string]string{
				"gitlab.app": "auto-devops-examples/minimal-ruby-app",
				"gitlab.env": "prod",
			}

			mergeStringMap(values, tc.Values)

			options := &helm.Options{
				SetValues:      values,
				KubectlOptions: k8s.NewKubectlOptions("", "", namespaceName),
			}

			output := helm.RenderTemplate(t, options, helmChartPath, tc.Release, []string{"templates/worker-deployment.yaml"})

			var deployments deploymentAppsV1List
			helm.UnmarshalK8SYaml(t, output, &deployments)

			require.Len(t, deployments.Items, len(tc.ExpectedDeployments))

			for i, expectedDeployment := range tc.ExpectedDeployments {
				deployment := deployments.Items[i]
				require.Equal(t, expectedDeployment.ExpectedName, deployment.Name)
				require.Len(t, deployment.Spec.Template.Spec.Containers, 1)
				require.Equal(t, expectedDeployment.ExpectedCmd, deployment.Spec.Template.Spec.Containers[0].Command)
				require.Equal(t, expectedDeployment.ExpectedLivenessProbe, deployment.Spec.Template.Spec.Containers[0].LivenessProbe)
				require.Equal(t, expectedDeployment.ExpectedReadinessProbe, deployment.Spec.Template.Spec.Containers[0].ReadinessProbe)
			}
		})
	}
}

func TestNetworkPolicyDeployment(t *testing.T) {
	releaseName := "network-policy-test"
	templates := []string{"templates/network-policy.yaml"}
	expectedLabels := map[string]string{
		"app":      releaseName,
		"chart":    chartName,
		"release":  releaseName,
		"heritage": "Helm",
	}

	tcs := []struct {
		name       string
		valueFiles []string
		values     map[string]string

		expectedErrorRegexp *regexp.Regexp

		meta        metav1.ObjectMeta
		podSelector metav1.LabelSelector
		policyTypes []netV1.PolicyType
		ingress     []netV1.NetworkPolicyIngressRule
		egress      []netV1.NetworkPolicyEgressRule
	}{
		{
			name:                "disabled by default",
			expectedErrorRegexp: regexp.MustCompile("Error: could not find template templates/network-policy.yaml in chart"),
		},
		{
			name:        "with default policy",
			values:      map[string]string{"networkPolicy.enabled": "true"},
			meta:        metav1.ObjectMeta{Name: releaseName + "-auto-deploy", Labels: expectedLabels},
			podSelector: metav1.LabelSelector{MatchLabels: map[string]string{}},
			ingress: []netV1.NetworkPolicyIngressRule{
				{
					From: []netV1.NetworkPolicyPeer{
						{PodSelector: &metav1.LabelSelector{MatchLabels: map[string]string{}}},
						{NamespaceSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"app.gitlab.com/managed_by": "gitlab"},
						}},
					},
				},
			},
		},
		{
			name:        "with custom policy",
			valueFiles:  []string{"./testdata/custom-policy.yaml"},
			meta:        metav1.ObjectMeta{Name: releaseName + "-auto-deploy", Labels: expectedLabels},
			podSelector: metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
			ingress: []netV1.NetworkPolicyIngressRule{
				{
					From: []netV1.NetworkPolicyPeer{
						{PodSelector: &metav1.LabelSelector{MatchLabels: map[string]string{}}},
						{NamespaceSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"name": "foo"},
						}},
					},
				},
			},
		},
		{
			name:        "with full spec policy",
			valueFiles:  []string{"./testdata/full-spec-policy.yaml"},
			meta:        metav1.ObjectMeta{Name: releaseName + "-auto-deploy", Labels: expectedLabels},
			podSelector: metav1.LabelSelector{MatchLabels: map[string]string{}},
			policyTypes: []netV1.PolicyType{"Ingress", "Egress"},
			ingress: []netV1.NetworkPolicyIngressRule{
				{
					From: []netV1.NetworkPolicyPeer{
						{PodSelector: &metav1.LabelSelector{MatchLabels: map[string]string{}}},
					},
				},
			},
			egress: []netV1.NetworkPolicyEgressRule{
				{
					To: []netV1.NetworkPolicyPeer{
						{NamespaceSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"name": "gitlab-managed-apps"},
						}},
					},
				},
			},
		},
	}

	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			opts := &helm.Options{
				ValuesFiles: tc.valueFiles,
				SetValues:   tc.values,
			}
			output, err := helm.RenderTemplateE(t, opts, helmChartPath, releaseName, templates)

			if tc.expectedErrorRegexp != nil {
				require.Regexp(t, tc.expectedErrorRegexp, err.Error())
				return
			}
			if err != nil {
				t.Error(err)
				return
			}

			policy := new(netV1.NetworkPolicy)
			helm.UnmarshalK8SYaml(t, output, policy)

			require.Equal(t, tc.meta, policy.ObjectMeta)
			require.Equal(t, tc.podSelector, policy.Spec.PodSelector)
			require.Equal(t, tc.policyTypes, policy.Spec.PolicyTypes)
			require.Equal(t, tc.ingress, policy.Spec.Ingress)
			require.Equal(t, tc.egress, policy.Spec.Egress)
		})
	}
}

func TestIngressTemplate_ModSecurity(t *testing.T) {
	templates := []string{"templates/ingress.yaml"}
	modSecuritySnippet := "SecRuleEngine DetectionOnly\n"
	modSecuritySnippetWithSecRules := modSecuritySnippet + `SecRule REQUEST_HEADERS:User-Agent \"scanner\" \"log,deny,id:107,status:403,msg:\'Scanner Identified\'\"
SecRule REQUEST_HEADERS:Content-Type \"text/plain\" \"log,deny,id:\'20010\',status:403,msg:\'Text plain not allowed\'\"
`
	defaultAnnotations := map[string]string{
		"kubernetes.io/ingress.class": "nginx",
		"kubernetes.io/tls-acme":      "true",
	}
	defaultModSecurityAnnotations := map[string]string{
		"nginx.ingress.kubernetes.io/modsecurity-transaction-id": "$server_name-$request_id",
	}
	modSecurityAnnotations := make(map[string]string)
	secRulesAnnotations := make(map[string]string)
	mergeStringMap(modSecurityAnnotations, defaultAnnotations)
	mergeStringMap(modSecurityAnnotations, defaultModSecurityAnnotations)
	mergeStringMap(secRulesAnnotations, defaultAnnotations)
	mergeStringMap(secRulesAnnotations, defaultModSecurityAnnotations)
	modSecurityAnnotations["nginx.ingress.kubernetes.io/modsecurity-snippet"] = modSecuritySnippet
	secRulesAnnotations["nginx.ingress.kubernetes.io/modsecurity-snippet"] = modSecuritySnippetWithSecRules

	tcs := []struct {
		name       string
		valueFiles []string
		values     map[string]string
		meta       metav1.ObjectMeta
	}{
		{
			name: "defaults",
			meta: metav1.ObjectMeta{Annotations: defaultAnnotations},
		},
		{
			name:   "with modSecurity enabled without custom secRules",
			values: map[string]string{"ingress.modSecurity.enabled": "true"},
			meta:   metav1.ObjectMeta{Annotations: modSecurityAnnotations},
		},
		{
			name:       "with custom secRules",
			valueFiles: []string{"./testdata/modsecurity-ingress.yaml"},
			meta:       metav1.ObjectMeta{Annotations: secRulesAnnotations},
		},
	}

	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			opts := &helm.Options{
				ValuesFiles: tc.valueFiles,
				SetValues:   tc.values,
			}
			output := helm.RenderTemplate(t, opts, helmChartPath, "ModSecurity-test-release", templates)

			ingress := new(extensions.Ingress)
			helm.UnmarshalK8SYaml(t, output, ingress)

			require.Equal(t, tc.meta.Annotations, ingress.ObjectMeta.Annotations)
		})
	}
}

func TestIngressTemplate_DifferentTracks(t *testing.T) {
	templates := []string{"templates/ingress.yaml"}
	tcs := []struct {
		name        string
		releaseName string
		values      map[string]string

		expectedName                     string
		expectedLabels                   map[string]string
		expectedSelector                 map[string]string
		expectedAnnotations              map[string]string
		expectedInexistentAnnotationKeys []string
		expectedErrorRegexp              *regexp.Regexp
	}{
		{
			name:                             "defaults",
			releaseName:                      "production",
			expectedName:                     "production-auto-deploy",
			expectedAnnotations:              map[string]string{"kubernetes.io/ingress.class": "nginx"},
			expectedInexistentAnnotationKeys: []string{"nginx.ingress.kubernetes.io/canary"},
		},
		{
			name:                             "with canary track",
			releaseName:                      "production-canary",
			values:                           map[string]string{"application.track": "canary"},
			expectedName:                     "production-canary-auto-deploy",
			expectedAnnotations:              map[string]string{"nginx.ingress.kubernetes.io/canary": "true", "nginx.ingress.kubernetes.io/canary-by-header": "canary", "kubernetes.io/ingress.class": "nginx"},
			expectedInexistentAnnotationKeys: []string{"nginx.ingress.kubernetes.io/canary-weight"},
		},
		{
			name:                "with canary weight",
			releaseName:         "production-canary",
			values:              map[string]string{"application.track": "canary", "ingress.canary.weight": "25"},
			expectedName:        "production-canary-auto-deploy",
			expectedAnnotations: map[string]string{"nginx.ingress.kubernetes.io/canary-weight": "25"},
		},
	}

	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			output, ret := renderTemplate(t, tc.values, tc.releaseName, templates, tc.expectedErrorRegexp)

			if ret == false {
				return
			}

			ingress := new(extensions.Ingress)
			helm.UnmarshalK8SYaml(t, output, ingress)
			require.Equal(t, tc.expectedName, ingress.ObjectMeta.Name)
			for key, value := range tc.expectedAnnotations {
				require.Equal(t, ingress.ObjectMeta.Annotations[key], value)
			}
			for _, key := range tc.expectedInexistentAnnotationKeys {
				require.Empty(t, ingress.ObjectMeta.Annotations[key])
			}
		})
	}
}

func TestIngressTemplate_Disable(t *testing.T) {
	templates := []string{"templates/ingress.yaml"}
	releaseName := "ingress-disable-test"
	tcs := []struct {
		name   string
		values map[string]string

		expectedName        string
		expectedErrorRegexp *regexp.Regexp
	}{
		{
			name:         "defaults",
			expectedName: releaseName + "-auto-deploy",
		},
		{
			name:         "with ingress.enabled key undefined, but service is enabled",
			values:       map[string]string{"ingress.enabled": "null", "service.enabled": "true"},
			expectedName: releaseName + "-auto-deploy",
		},
		{
			name:                "with service disabled and track stable",
			values:              map[string]string{"service.enabled": "false", "application.track": "stable"},
			expectedErrorRegexp: regexp.MustCompile("Error: could not find template templates/ingress.yaml in chart"),
		},
		{
			name:                "with service disabled and track non-stable",
			values:              map[string]string{"service.enabled": "false", "application.track": "non-stable"},
			expectedErrorRegexp: regexp.MustCompile("Error: could not find template templates/ingress.yaml in chart"),
		},
		{
			name:                "with ingress disabled",
			values:              map[string]string{"ingress.enabled": "false"},
			expectedErrorRegexp: regexp.MustCompile("Error: could not find template templates/ingress.yaml in chart"),
		},
		{
			name:                "with ingress enabled and service disabled",
			values:              map[string]string{"ingress.enabled": "true", "service.enabled": "false"},
			expectedErrorRegexp: regexp.MustCompile("Error: could not find template templates/ingress.yaml in chart"),
		},
		{
			name:                "with ingress disabled and service enabled and track stable",
			values:              map[string]string{"ingress.enabled": "false", "service.enabled": "true", "application.track": "stable"},
			expectedErrorRegexp: regexp.MustCompile("Error: could not find template templates/ingress.yaml in chart"),
		},
	}

	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			opts := &helm.Options{
				SetValues: tc.values,
			}
			output, err := helm.RenderTemplateE(t, opts, helmChartPath, releaseName, templates)

			if tc.expectedErrorRegexp != nil {
				require.Regexp(t, tc.expectedErrorRegexp, err.Error())
				return
			}
			if err != nil {
				t.Error(err)
				return
			}

			ingress := new(extensions.Ingress)
			helm.UnmarshalK8SYaml(t, output, ingress)
			require.Equal(t, tc.expectedName, ingress.ObjectMeta.Name)
		})
	}
}

func TestServiceTemplate_DifferentTracks(t *testing.T) {
	templates := []string{"templates/service.yaml"}
	tcs := []struct {
		name        string
		releaseName string
		values      map[string]string

		expectedName        string
		expectedLabels      map[string]string
		expectedSelector    map[string]string
		expectedErrorRegexp *regexp.Regexp
	}{
		{
			name:             "defaults",
			releaseName:      "production",
			expectedName:     "production-auto-deploy",
			expectedLabels:   map[string]string{"app": "production", "release": "production", "track": "stable"},
			expectedSelector: map[string]string{"app": "production", "tier": "web", "track": "stable"},
		},
		{
			name:             "with canary track",
			releaseName:      "production-canary",
			values:           map[string]string{"application.track": "canary"},
			expectedName:     "production-canary-auto-deploy",
			expectedLabels:   map[string]string{"app": "production-canary", "release": "production-canary", "track": "canary"},
			expectedSelector: map[string]string{"app": "production-canary", "tier": "web", "track": "canary"},
		},
	}

	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			output, ret := renderTemplate(t, tc.values, tc.releaseName, templates, tc.expectedErrorRegexp)

			if ret == false {
				return
			}

			service := new(coreV1.Service)
			helm.UnmarshalK8SYaml(t, output, service)
			require.Equal(t, tc.expectedName, service.ObjectMeta.Name)
			for key, value := range tc.expectedLabels {
				require.Equal(t, service.ObjectMeta.Labels[key], value)
			}
			for key, value := range tc.expectedSelector {
				require.Equal(t, service.Spec.Selector[key], value)
			}
		})
	}
}

func TestServiceTemplate_Disable(t *testing.T) {
	templates := []string{"templates/service.yaml"}
	releaseName := "service-disable-test"
	tcs := []struct {
		name   string
		values map[string]string

		expectedName        string
		expectedErrorRegexp *regexp.Regexp
	}{
		{
			name:         "defaults",
			expectedName: releaseName + "-auto-deploy",
		},
		{
			name:                "with service disabled and track stable",
			values:              map[string]string{"service.enabled": "false", "application.track": "stable"},
			expectedErrorRegexp: regexp.MustCompile("Error: could not find template templates/service.yaml in chart"),
		},
		{
			name:                "with service disabled and track non-stable",
			values:              map[string]string{"service.enabled": "false", "application.track": "non-stable"},
			expectedErrorRegexp: regexp.MustCompile("Error: could not find template templates/service.yaml in chart"),
		},
	}

	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			output, ret := renderTemplate(t, tc.values, releaseName, templates, tc.expectedErrorRegexp)

			if ret == false {
				return
			}

			service := new(coreV1.Service)
			helm.UnmarshalK8SYaml(t, output, service)
			require.Equal(t, tc.expectedName, service.ObjectMeta.Name)
		})
	}
}

func renderTemplate(t *testing.T, values map[string]string, releaseName string, templates []string, expectedErrorRegexp *regexp.Regexp) (string, bool) {
	opts := &helm.Options{
		SetValues: values,
	}

	output, err := helm.RenderTemplateE(t, opts, helmChartPath, releaseName, templates)
	if expectedErrorRegexp != nil {
		if err == nil {
			t.Error("Expected error but didn't happen")
		} else {
			require.Regexp(t, expectedErrorRegexp, err.Error())
		}
		return "", false
	}
	if err != nil {
		t.Error(err)
		return "", false
	}

	return output, true
}

type workerDeploymentTestCase struct {
	ExpectedName           string
	ExpectedCmd            []string
	ExpectedStrategyType   appsV1.DeploymentStrategyType
	ExpectedSelector       *metav1.LabelSelector
	ExpectedLivenessProbe  *coreV1.Probe
	ExpectedReadinessProbe *coreV1.Probe
}

type workerDeploymentSelectorTestCase struct {
	ExpectedName     string
	ExpectedSelector *metav1.LabelSelector
}

type deploymentList struct {
	metav1.TypeMeta `json:",inline"`

	Items []appsV1.Deployment `json:"items" protobuf:"bytes,2,rep,name=items"`
}

type deploymentAppsV1List struct {
	metav1.TypeMeta `json:",inline"`

	Items []appsV1.Deployment `json:"items" protobuf:"bytes,2,rep,name=items"`
}

func mergeStringMap(dst, src map[string]string) {
	for k, v := range src {
		dst[k] = v
	}
}

func defaultLivenessProbe() *coreV1.Probe {
	return &coreV1.Probe{
		Handler: coreV1.Handler{
			HTTPGet: &coreV1.HTTPGetAction{
				Path:   "/",
				Port:   intstr.FromInt(5000),
				Scheme: coreV1.URISchemeHTTP,
			},
		},
		InitialDelaySeconds: 15,
		TimeoutSeconds:      15,
	}
}

func defaultReadinessProbe() *coreV1.Probe {
	return &coreV1.Probe{
		Handler: coreV1.Handler{
			HTTPGet: &coreV1.HTTPGetAction{
				Path:   "/",
				Port:   intstr.FromInt(5000),
				Scheme: coreV1.URISchemeHTTP,
			},
		},
		InitialDelaySeconds: 5,
		TimeoutSeconds:      3,
	}
}

func workerLivenessProbe() *coreV1.Probe {
	return &coreV1.Probe{
		Handler: coreV1.Handler{
			HTTPGet: &coreV1.HTTPGetAction{
				Path:   "/worker",
				Port:   intstr.FromInt(5000),
				Scheme: coreV1.URISchemeHTTP,
			},
		},
		InitialDelaySeconds: 0,
		TimeoutSeconds:      0,
	}
}

func workerReadinessProbe() *coreV1.Probe {
	return &coreV1.Probe{
		Handler: coreV1.Handler{
			HTTPGet: &coreV1.HTTPGetAction{
				Path:   "/worker",
				Port:   intstr.FromInt(5000),
				Scheme: coreV1.URISchemeHTTP,
			},
		},
		InitialDelaySeconds: 0,
		TimeoutSeconds:      0,
	}
}
